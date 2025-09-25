import os
import gc
import time
import comfy
import io
import torch
import json
import nodes
import folder_paths

from .gguf_connector import reader as gr
from .gguf_connector.writer import GGUFWriter, GGMLQuantizationType
from .gguf_connector.const import GGML_QUANT_VERSION, LlamaFileType
from .gguf_connector.quant import quantize, dequantize, QuantError
from .gguf_connector.quant2d import dequantize_tensor, is_quantized, is_torch_compatible
from .gguf_connector.tkn import get_field, tokenizer_builder

def get_available_devices():
    available_devices = ["default", "cpu"]

    if torch.cuda.is_available():
        available_devices.append("cuda")

        if torch.cuda.device_count() > 1:
            for i in range(torch.cuda.device_count()):
                temp_device = "cuda:{}".format(i)
                available_devices.append(temp_device)

    return available_devices

def get_offload_device():
    return torch.device("cpu")

if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'disable'):
    torch_compiler_disable = torch.compiler.disable
else:
    def torch_compiler_disable(*args, **kwargs):
        def noop(x):
            return x
        return noop

def get_clip_type(type):
    clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
    return clip_type

def get_device(device):
    model_options = {}

    print("get_device: input device: {}".format(device))

    if device == "cpu":
        model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
    elif device.startswith("cuda"):
        model_options["load_device"] = torch.device(device)
        model_options["offload_device"] = torch.device("cpu")

    print("get_device: model_options: {}".format(model_options))

    return model_options

def tensor_swap(raw_sd, key_map):
    sd = {}
    for k, v in raw_sd.items():
        for s, d in key_map.items():
            k = k.replace(s, d)
        sd[k] = v
    return sd

def llama_permute(raw_sd, n_head, n_head_kv):
    sd = {}
    permute = lambda x, h: x.reshape(h, x.shape[0] // h // 2, 2, *x.shape[1:]
        ).swapaxes(1, 2).reshape(x.shape)
    for k, v in raw_sd.items():
        if k.endswith(('q_proj.weight', 'q_proj.bias')):
            v.data = permute(v.data, n_head)
        if k.endswith(('k_proj.weight', 'k_proj.bias')):
            v.data = permute(v.data, n_head_kv)
        sd[k] = v
    return sd

def load_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(load_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [load_patch_to_device(x, device) for x in item]
    else:
        return item

class GGMLTensor(torch.Tensor):
    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches
    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, *args, **kwargs)
    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, 'tensor_type', None)
        new.tensor_shape = getattr(self, 'tensor_shape', new.data.shape)
        new.patches = getattr(self, 'patches', []).copy()
        return new
    def clone(self, *args, **kwargs):
        return self
    def detach(self, *args, **kwargs):
        return self
    def copy_(self, *args, **kwargs):
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            print(f"ignoring 'copy_' on tensor: {e}")
    def empty_(self, size, *args, **kwargs):
        new_tensor = super().empty_(size, *args, **kwargs)
        return GGMLTensor(new_tensor, tensor_type=getattr(self,
            'tensor_type', None), tensor_shape=size, patches=getattr(self,
            'patches', []).copy())
    @property
    def shape(self):
        if not hasattr(self, 'tensor_shape'):
            self.tensor_shape = self.size()
        return self.tensor_shape

class GGMLLayer(torch.nn.Module):
    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    largest_layer = False
    torch_compatible_tensor_types = {None, gr.GGMLQuantizationType.F32, gr.
        GGMLQuantizationType.F16}
    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f'{prefix}weight'), state_dict.get(
            f'{prefix}bias')
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self,
            torch.nn.Linear):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args,
                **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **
            kwargs)
    def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k[prefix_len:] == 'weight':
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == 'bias' and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                missing_keys.append(k)
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix + 'weight')
        if getattr(self.weight, 'is_largest_weight', False):
            self.largest_layer = True
    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)
    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        weight = torch.zeros_like(self.weight, device=torch.device('meta'))
        destination[prefix + 'weight'] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device('meta'))
            destination[prefix + 'bias'] = bias
        if self.largest_layer:
            shape = getattr(self.weight, 'tensor_shape', self.weight.shape)
            dtype = self.dequant_dtype or torch.float16
            temp = torch.empty(*shape, device=torch.device('meta'), dtype=dtype
                )
            destination[prefix + 'temp.weight'] = temp
        return
        destination[prefix + 'weight'] = self.get_weight(self.weight)
        if bias is not None:
            destination[prefix + 'bias'] = self.get_weight(self.bias)
    def get_weight(self, tensor, dtype):
        if tensor is None:
            return
        patch_list = []
        device = tensor.device
        for function, patches, key in getattr(tensor, 'patches', []):
            patch_list += load_patch_to_device(patches, device)
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)
        if patch_list:
            if self.patch_dtype is None:
                weight = function(patch_list, weight, key)
            else:
                patch_dtype = (dtype if self.patch_dtype == 'target' else
                    self.patch_dtype)
                weight = function(patch_list, weight, key, patch_dtype)
        return weight
    @torch_compiler_disable()
    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype
        =None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, 'dtype', torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device
        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(
            device)
        if s.bias is not None:
            bias = s.get_weight(s.bias.to(device), dtype)
            bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking
                =non_blocking, copy=False)
        weight = s.get_weight(s.weight.to(device), dtype)
        weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=
            non_blocking, copy=False)
        return weight, bias
    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)
        return out
    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError

class GGMLOps(comfy.ops.manual_cast):
    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=
            None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)
    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)
    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if (self.weight.dtype == torch.float16 or self.weight.dtype ==
                torch.bfloat16):
                out_dtype = None
            weight, _bias = self.cast_bias_weight(self, device=input.device,
                dtype=out_dtype)
            return torch.nn.functional.embedding(input, weight, self.
                padding_idx, self.max_norm, self.norm_type, self.
                scale_grad_by_freq, self.sparse).to(dtype=output_dtype)
    class LayerNorm(GGMLLayer, comfy.ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(input, self.
                normalized_shape, weight, bias, self.eps)
    class GroupNorm(GGMLLayer, comfy.ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(input, self.num_groups,
                weight, bias, self.eps)

pig = os.path.join(os.path.dirname(__file__), 'version.json')
with open(pig, 'r') as file:
    data = json.load(file)
arrays = {}
for key, value in data[0].items():
    arrays[key] = value

def get_orig_shape(reader, tensor_name):
    field_key = f'comfy.gguf.orig_shape.{tensor_name}'
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0
        ] != gr.GGUFValueType.ARRAY or field.types[1
        ] != gr.GGUFValueType.INT32:
        raise TypeError(
            f'Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}'
            )
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in
        field.data))

class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False
    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)
        try:
            from comfy.lora import calculate_weight
        except Exception:
            calculate_weight = self.calculate_weight
        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = load_patch_to_device(patches, self.load_device if
                self.patch_on_device else self.offload_device)
            out_weight.patches = [(calculate_weight, patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', [
                    'weight', 'inplace_update'])(weight.to(device=self.
                    offload_device, copy=inplace_update), inplace_update)
            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight,
                    device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)
            out_weight = calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight
                .dtype)
        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)
    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, 'patches', [])
                if len(patches) > 0:
                    p.patches = []
        return super().unpatch_model(device_to=device_to, unpatch_weights=
            unpatch_weights)
    mmap_released = False
    def load(self, *args, force_patch_weights=False, **kwargs):
        super().load(*args, force_patch_weights=True, **kwargs)
        if not self.mmap_released:
            linked = []
            if kwargs.get('lowvram_model_memory', 0) > 0:
                for n, m in self.model.named_modules():
                    if hasattr(m, 'weight'):
                        device = getattr(m.weight, 'device', None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, 'bias'):
                        device = getattr(m.bias, 'device', None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked:
                print(f'Attempting to release mmap ({len(linked)})')
                for n, m in linked:
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True
    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        n.patch_on_device = getattr(self, 'patch_on_device', False)
        return n

def load_gguf_sd(path, handle_prefix='model.diffusion_model.', return_arch=
    False):
    reader = gr.GGUFReader(path)
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)
    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # print(arrays)

    compat = None
    arch_str = get_field(reader, 'general.architecture', str)

    # print("arch_str : {}".format(arch_str))

    if arch_str is None:
        compat = 'sd.cpp'
    elif arch_str not in arrays['PIG_ARCH_LIST'] and arch_str not in arrays['TXT_ARCH_LIST']:
        raise ValueError(f"Unknown architecture: {arch_str!r}")
    state_dict, qtype_dict = {}, {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        torch_tensor = torch.from_numpy(tensor.data)
        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            if compat == 'sd.cpp' and arch_str == 'sdxl':
                if any([tensor_name.endswith(x) for x in ('.proj_in.weight',
                    '.proj_out.weight')]):
                    while len(shape) > 2 and shape[-1] == 1:
                        shape = shape[:-1]
        if tensor.tensor_type in {gr.GGMLQuantizationType.F32, gr.
            GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.
            tensor_type, tensor_shape=shape)
        tensor_type_str = getattr(tensor.tensor_type, 'name', repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1
    print('gguf qtypes: ' + ', '.join(f'{k} ({v})' for k, v in qtype_dict.items()))
    qsd = {k: v for k, v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True
    if return_arch:
        return state_dict, arch_str
    return state_dict

class WarpedLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        gguf_names = [x for x in folder_paths.get_filename_list('model_gguf')]
        return {'required': {'gguf_name': (gguf_names,)}}
    RETURN_TYPES = 'MODEL',
    FUNCTION = 'load_model'
    CATEGORY = 'Warped/GGUF/Loaders'

    def load_model(self, gguf_name, dequant_dtype=None, patch_dtype=None,
        patch_on_device=None):
        ops = GGMLOps()
        if dequant_dtype in ('default', None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ['target']:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)
        if patch_dtype in ('default', None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ['target']:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)
        model_path = folder_paths.get_full_path('unet', gguf_name)

        print("Reading Checkpoint: {}...".format(model_path))
        with open(model_path, "rb") as file:
            checkpoint_temp = file.read()
        print("Reading Checkpoint: {}...Done!".format(model_path))

        print("Loading Checkpoint...")

        temp_io = io.BytesIO(checkpoint_temp)

        sd = load_gguf_sd(temp_io)
        print("Loading Checkpoint...Done!")

        # print("\n")
        # for key in sd.keys():
        #     print(key)
        # print("\n")

        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=
            {'custom_operations': ops})
        if model is None:
            logging.error('ERROR UNSUPPORTED MODEL {}'.format(model_path))
            raise RuntimeError('ERROR: Could not detect model type of: {}'.
                format(model_path))

        # print("\n")
        # print(model.model)
        # print("\n")

        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device

        temp_io = None
        checkpoint_temp = None
        gc.collect()
        time.sleep(1)

        return model,

class WarpedWan22MergeLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        gguf_names = [x for x in folder_paths.get_filename_list('model_gguf')]
        return {'required': {
                   "model1": (gguf_names,),
                   "strength1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                   "model2": (gguf_names,),
                   "strength2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                   }
               }

    RETURN_TYPES = ("MODEL", )
    RETURN_NAMES = ("model", )
    FUNCTION = 'do_merge'
    CATEGORY = 'Warped/GGUF/Loaders'

    def load_state_dict(self, gguf_name, dequant_dtype=None):
        ops = GGMLOps()
        if dequant_dtype in ('default', None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ['target']:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)
        if patch_dtype in ('default', None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ['target']:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)
        model_path = folder_paths.get_full_path('unet', gguf_name)

        print("Reading Checkpoint: {}...".format(model_path))
        with open(model_path, "rb") as file:
            checkpoint_temp = file.read()
        print("Reading Checkpoint: {}...Done!".format(model_path))

        print("Loading Checkpoint...")

        temp_io = io.BytesIO(checkpoint_temp)

        sd = load_gguf_sd(temp_io)
        print("Loading Checkpoint...Done!")

        print("\ngguf_name: {}\n".format(gguf_name))
        for key in sd.keys():
            print(key)
        print("\n")

        temp_io = None
        checkpoint_temp = None
        gc.collect()
        time.sleep(1)

        return sd

    def do_merge(self, model1, strength1, model2, strength2, dequant_dtype=None):
        state_dict1 = self.load_state_dict(self, gguf_name, dequant_dtype=dequant_dtype)
        state_dict2 = self.load_state_dict(self, gguf_name, dequant_dtype=dequant_dtype)

        new_state_dict = {}

        for key in state_dict1.keys():
            temp_weights1 = torch.mul(state_dict1[key], strength1)

            if key in temp_weights.keys():
                temp_weights2 = torch.mul(state_dict2[key], strength2)
                new_state_dict[key] = torch.add(temp_weights1, temp_weights2)
            else:
                new_state_dict[key] = temp_weights1

        for key in state_dict2.keys():
            if key in new_state_dict.keys():
                continue

            new_state_dict[key] = state_dict2[key]

        model = comfy.sd.load_diffusion_model_state_dict(new_state_dict, model_options={'custom_operations': ops})

        if model is None:
            logging.error('ERROR UNSUPPORTED MODEL {}'.format(model_path))
            raise RuntimeError('ERROR: Could not detect model type of: {}'.
                format(model_path))

        # print("\n")
        # print(model.model)
        # print("\n")

        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device

        temp_io = None
        checkpoint_temp = None
        gc.collect()
        time.sleep(1)

        return model,

def warped_load_torch_file(ckpt, return_metadata=False):
    from safetensors.torch import load as safeload

    metadata = None

    try:
        f = safeload(ckpt)

        sd = {}
        for k in f.keys():
            if k != "__metadata__":
                sd[k] = f[k]
            elif return_metadata:
                metadata = f[k]
    except Exception as e:
        if len(e.args) > 0:
            message = e.args[0]
            if "HeaderTooLarge" in message:
                raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt or invalid. Make sure this is actually a safetensors file and not a ckpt or pt or other filetype.".format(message, ckpt))
            if "MetadataIncompleteBuffer" in message:
                raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt/incomplete. Check the file size and make sure you have copied/downloaded it correctly.".format(message, ckpt))
        raise e

    return (sd, metadata) if return_metadata else sd

def load_gguf_clip(path):
    print("Reading Clip: {}...".format(path))
    with open(path, "rb") as file:
        checkpoint_temp = file.read()
    print("Reading Clip: {}...Done!".format(path))

    print("Loading Clip...")
    # temp = warped_load_torch_file(checkpoint_temp)
    temp_io = io.BytesIO(checkpoint_temp)
    # print("temp_io Type: {}".format(type(temp_io)))

    sd, arch = load_gguf_sd(temp_io, return_arch=True)
    print("Loading Clip...Done!")

    if arch in {'t5', 't5encoder'}:
        temb_key = 'token_embd.weight'
        if temb_key in sd and (sd[temb_key].shape == (256384, 4096) or sd[temb_key].shape == (256384, 768)):
            sd['spiece_model'] = tokenizer_builder(path)
            sd = tensor_swap(sd, arrays['T5'])
        elif temb_key in sd and sd[temb_key].shape == (32128, 768):
            sd = tensor_swap(sd, arrays['B5'])
        else:
            sd = tensor_swap(sd, arrays['T5'])
    elif arch in {'llama'}:
        sd = tensor_swap(sd, arrays['L3'])
        sd = llama_permute(sd, 32, 8)
    elif arch in {'gemma2'}:
        sd["spiece_model"] = tokenizer_builder(path)
        sd = tensor_swap(sd, arrays['G2'])
    elif arch in {'pig'}:
        sd = pig_work(sd)
    else:
        pass

    temp_io = None

    return sd

class WarpedClipLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.CLIPLoader.INPUT_TYPES()
        return {'required': {
                            'clip_name': (s.get_filename_list(),),
                            'type': base['required']['type']
                            },
                'optional': {
                            'device':(get_available_devices(),{'advanced':True}),
                            }
               }
    RETURN_TYPES = 'CLIP',
    FUNCTION = 'load_clip'
    CATEGORY = 'Warped/GGUF/Loaders'

    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list('clip')
        files += folder_paths.get_filename_list('clip_gguf')
        return sorted(files)

    def load_data(self, ckpt_paths):
        clip_data = []
        for p in ckpt_paths:
            if p.endswith('.gguf'):
                sd = load_gguf_clip(p)
            else:
                print("Reading Clip: {}...".format(p))
                with open(p, "rb") as file:
                    checkpoint_temp = file.read()
                print("Reading Clip: {}...Done!".format(p))
                sd = warped_load_torch_file(checkpoint_temp)

            clip_data.append(sd)
        return clip_data

    def load_patcher(self, clip_paths, clip_type, clip_data):
        clip = comfy.sd.load_text_encoder_state_dicts(clip_type=clip_type,
            state_dicts=clip_data, model_options={'custom_operations':
            GGMLOps, 'initial_device': comfy.model_management.
            text_encoder_offload_device()}, embedding_directory=
            folder_paths.get_folder_paths('embeddings'))
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)
        return clip

    def sd_load_clip(self, ckpt_paths, embedding_directory=None, clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options={"offload_device": get_offload_device()}):
        checkpoint_temp = []
        for p in ckpt_paths:
            print("Reading: {}".format(p))
            with open(p, "rb") as file:
                content = file.read()

            checkpoint_temp.append(content)

        clip_data = []

        for p in checkpoint_temp:
            print("Loading Clip...")
            clip_data.append(warped_load_torch_file(p))
            print("Loading Clip...Done")

        return_clip = comfy.sd.load_text_encoder_state_dicts(clip_data, embedding_directory=embedding_directory, clip_type=clip_type, model_options=model_options)

        checkpoint_temp = None
        clip_data = None

        return return_clip

    def load_clip(self, clip_name, clip_type="stable_diffusion", device="cpu"):
        clip_type_attr = getattr(comfy.sd.CLIPType, clip_type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

        print("WarpedCLIPLoader: Clip Type: {}  |  {}".format(clip_type, clip_type_attr))

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        else:
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

            if torch.cuda.is_available():
                if device == "cuda":
                    model_options["load_device"] = torch.cuda.current_device()
                else:
                    temp_device = device.split(':')
                    device_number = int(temp_device[len(temp_device) - 1])
                    model_options["load_device"] = torch.device(device_number)

        print("WarpedCLIPLoader: {}".format(model_options))

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip = self.sd_load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type_attr, model_options=model_options)
        return (clip,)

    # def load_clip(self, clip_name, type='stable_diffusion', device='default'):
    #     clip_path = folder_paths.get_full_path('clip', clip_name)
    #     if clip_name.endswith('.safetensors') and device != 'default':
    #         clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=get_clip_type(type), model_options=get_device(device))
    #         return (clip,)
    #     else:
    #         return (self.load_patcher([clip_path], get_clip_type(type), self.load_data([clip_path])), get_device(device))

class WarpedDualClipLoaderGGUF(WarpedClipLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.DualCLIPLoader.INPUT_TYPES()
        file_options = s.get_filename_list(),
        return {'required': {
                            'clip_name1':file_options,
                            'clip_name2':file_options,
                            'type': base['required']['type']
                            },
               'optional': {
                            'device':(get_available_devices(),{'advanced':True}),
                           }
               }

    def load_clip(self, clip_name1, clip_name2, type, device='default'):
        clip_path1 = folder_paths.get_full_path('clip', clip_name1)
        clip_path2 = folder_paths.get_full_path('clip', clip_name2)
        clip_paths = [clip_path1, clip_path2]
        if device != 'default' and (clip_name1.endswith('.safetensors') and clip_name2.endswith('.safetensors')):
            clip = self.load_clip(ckpt_paths=clip_paths, embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=get_clip_type(type), model_options=get_device(device))
            # clip = comfy.sd.load_clip(ckpt_paths=clip_paths, embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=get_clip_type(type), model_options=get_device(device))
            return (clip,)
        else:
            return (self.load_patcher(clip_paths, get_clip_type(type), self.load_data(clip_paths)), get_device(device))
