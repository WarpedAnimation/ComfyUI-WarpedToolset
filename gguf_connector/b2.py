import os
if not os.path.isfile(os.path.join(os.path.dirname(__file__),
    'models/bagel/config.json')):
    from huggingface_hub import snapshot_download
    save_dir = os.path.join(os.path.dirname(__file__), 'models/bagel')
    repo_id = 'callgg/bagel-decoder'
    cache_dir = save_dir + '/cache'
    snapshot_download(cache_dir=cache_dir, local_dir=save_dir, repo_id=
        repo_id, allow_patterns=['*.json', '*.md', '*.txt'])
import time, psutil, platform, atexit, random
pynvml_available = False
if platform.system() == 'Linux' or platform.system() == 'Windows':
    try:
        from pynvml import *
        nvmlInit()
        pynvml_available = True
        print(
            """pynvml (NVIDIA GPU monitoring library) initialized successfully!
Detecting GGUF/Safetensors...
"""
            )
        def shutdown_pynvml():
            print('Shutting down pynvml...')
            nvmlShutdown()
        atexit.register(shutdown_pynvml)
    except Exception as e:
        print(
            f'Warning: pynvml could not be initialized. Detailed GPU stats via pynvml will not be available. Error: {e}'
            )
        if 'NVML Shared Library Not Found' in str(e):
            print(
                'pynvml error hint: NVML shared library not found. If you have an NVIDIA GPU and drivers, ensure the library is accessible.'
                )
        elif 'Driver Not Loaded' in str(e):
            print(
                'pynvml error hint: NVIDIA driver is not loaded. Please check your GPU driver installation.'
                )
import torch
import gradio as gr
import numpy as np
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from PIL import Image
from bagel2.data.data_utils import add_special_tokens, pil_img2rgb
from bagel2.data.transforms import ImageTransform
from bagel2.inferencer import InterleaveInferencer
from bagel2.modeling.autoencoder import load_ae
from bagel2.modeling.bagel.qwen2_navit import NaiveCache
from bagel2.modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from bagel2.modeling.qwen2 import Qwen2Tokenizer
model_path = os.path.join(os.path.dirname(__file__), 'models/bagel')
llm_config = Qwen2Config.from_json_file(os.path.join(model_path,
    'llm_config.json'))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = 'Qwen2MoTDecoderLayer'
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path,
    'vit_config.json'))
vit_config.rope = False
vit_config.num_hidden_layers -= 1
from safetensors.torch import save_file
from typing import Dict, Tuple
from tqdm import tqdm
from .reader import GGUFReader
from .quant import dequantize
def load_gguf_and_extract_metadata(gguf_path):
    reader = GGUFReader(gguf_path)
    tensors_metadata = []
    for tensor in reader.tensors:
        tensor_metadata = {'name': tensor.name, 'shape': tuple(tensor.shape
            .tolist()), 'n_elements': tensor.n_elements, 'n_bytes': tensor.
            n_bytes, 'data_offset': tensor.data_offset, 'type': tensor.
            tensor_type}
        tensors_metadata.append(tensor_metadata)
    return reader, tensors_metadata
def convert_gguf_to_safetensors(gguf_path, output_path, use_bf16):
    reader, tensors_metadata = load_gguf_and_extract_metadata(gguf_path)
    print(f'Extracted {len(tensors_metadata)} tensors from GGUF file')
    tensors_dict: dict[str, torch.Tensor] = {}
    for i, tensor_info in enumerate(tqdm(tensors_metadata, desc=
        'Dequantizing tensors', unit='tensor')):
        tensor_name = tensor_info['name']
        tensor_data = reader.get_tensor(i)
        weights = dequantize(tensor_data.data, tensor_data.tensor_type).copy()
        try:
            if use_bf16:
                weights_tensor = torch.from_numpy(weights).to(dtype=torch.
                    float32)
                weights_tensor = weights_tensor.to(torch.bfloat16)
            else:
                weights_tensor = torch.from_numpy(weights).to(dtype=torch.
                    float32)
            weights_hf = weights_tensor
        except Exception as e:
            print(
                f"Error during dequantization for tensor '{tensor_name}': {e}")
            weights_tensor = torch.from_numpy(weights.astype(np.float32)).to(
                torch.float16)
            weights_hf = weights_tensor
        tensors_dict[tensor_name] = weights_hf
    metadata = {key: str(reader.get_field(key)) for key in reader.fields}
    save_file(tensors_dict, output_path, metadata=metadata)
    print('Dequantization complete!')
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]
safetensors_files = [file for file in os.listdir() if file.endswith(
    '.safetensors')]
if gguf_files:
    print('GGUF file(s) available. Select which one for VAE:')
    for index, file_name in enumerate(gguf_files, start=1):
        print(f'{index}. {file_name}')
    choice = input(f'Enter your choice (1 to {len(gguf_files)}): ')
    try:
        choice_index = int(choice) - 1
        selected_file = gguf_files[choice_index]
        print(f'VAE file: {selected_file} is selected!')
        input_path = selected_file
        use_bf16 = False
        vae_path = f'{os.path.splitext(input_path)[0]}-f32.safetensors'
        if safetensors_files:
            print(
                '\nSafetensors file(s) available. Select which one for MODEL:')
            for index, file_name in enumerate(safetensors_files, start=1):
                print(f'{index}. {file_name}')
            choice2 = input(
                f'Enter your choice (1 to {len(safetensors_files)}): ')
            try:
                choice_index = int(choice2) - 1
                selected_model_file = safetensors_files[choice_index]
                print(f'MODEL file: {selected_model_file} is selected!\n')
                convert_gguf_to_safetensors(input_path, vae_path, use_bf16)
                vae_model, vae_config = load_ae(vae_path)
                checkpoint = selected_model_file
            except (ValueError, IndexError):
                print('Invalid choice. Please enter a valid number.')
    except (ValueError, IndexError):
        print('Invalid choice. Please enter a valid number.')
else:
    print('No GGUF/Safetensors are available in the current directory.')
    input('--- Press ENTER To Exit ---')
config = BagelConfig(visual_gen=True, visual_und=True, llm_config=
    llm_config, vit_config=vit_config, vae_config=vae_config,
    vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh',
    latent_patch_size=2, max_latent_size=64)
with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config
        , meta=True)
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)
print('Starting model loading and device map configuration...')
def get_gpu_memory_stats_pynvml(device_id=0):
    if not pynvml_available:
        return f'GPU-{device_id} (pynvml): Not available.'
    try:
        handle = nvmlDeviceGetHandleByIndex(device_id)
        info = nvmlDeviceGetMemoryInfo(handle)
        total_gb = info.total / 1024 ** 3
        used_gb = info.used / 1024 ** 3
        return (
            f'GPU-{device_id} (pynvml): Total: {total_gb:.2f} GB, Used (Overall): {used_gb:.2f} GB'
            )
    except NVMLError as e:
        return f'GPU-{device_id} (pynvml) Error: {e}'
def get_gpu_memory_stats_pytorch(device_id=0):
    if not torch.cuda.is_available():
        return 'PyTorch: CUDA not available.'
    if device_id < 0 or device_id >= torch.cuda.device_count():
        return f'PyTorch GPU-{device_id}: Invalid device ID.'
    allocated_gb = torch.cuda.memory_allocated(device_id) / 1024 ** 3
    reserved_gb = torch.cuda.memory_reserved(device_id) / 1024 ** 3
    total_capacity_str_pt = ''
    if pynvml_available:
        try:
            handle = nvmlDeviceGetHandleByIndex(device_id)
            info = nvmlDeviceGetMemoryInfo(handle)
            total_gb_pt = info.total / 1024 ** 3
            total_capacity_str_pt = f'Total Capacity: {total_gb_pt:.2f} GB, '
        except NVMLError:
            pass
    return (
        f'PyTorch GPU-{device_id}: {total_capacity_str_pt}Allocated: {allocated_gb:.2f} GB, Reserved: {reserved_gb:.2f} GB'
        )
def get_system_ram_stats():
    mem = psutil.virtual_memory()
    total_gb = mem.total / 1024 ** 3
    available_gb = mem.available / 1024 ** 3
    used_gb = mem.used / 1024 ** 3
    percent_used = mem.percent
    return (
        f'System RAM: Total: {total_gb:.2f} GB, Available: {available_gb:.2f} GB, Used (Overall): {used_gb:.2f} GB ({percent_used}%)'
        )
def get_process_ram_stats():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / 1024 ** 3
    return f'App Process RAM (RSS): {rss_gb:.2f} GB'
def get_all_memory_stats_for_gradio_display():
    """Prepare the string of memory/video memory statistics for Gradio display"""
    stats_lines = []
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        stats_lines.append('**GPU VRAM Usage:**')
        for i in range(torch.cuda.device_count()):
            stats_lines.append(get_gpu_memory_stats_pynvml(i))
            stats_lines.append(get_gpu_memory_stats_pytorch(i))
            if i < torch.cuda.device_count() - 1:
                stats_lines.append('---')
    else:
        stats_lines.append(
            '**GPU VRAM Usage:** CUDA not available or no GPUs found.')
    stats_lines.append('\n**CPU RAM Usage:**')
    stats_lines.append(get_system_ram_stats())
    stats_lines.append(get_process_ram_stats())
    return '\n'.join(s for s in stats_lines if s)
cpu_mem_for_offload = '16GiB'
gpu_mem_per_device = '24GiB'
max_memory_config = {i: gpu_mem_per_device for i in range(torch.cuda.
    device_count())}
if torch.cuda.device_count() == 0:
    max_memory_config['cpu'] = cpu_mem_for_offload
else:
    max_memory_config['cpu'] = cpu_mem_for_offload
print(f'Using max_memory_config: {max_memory_config}')
device_map = infer_auto_device_map(model, max_memory=max_memory_config,
    no_split_module_classes=['Bagel', 'Qwen2MoTDecoderLayer'])
print('Device map after infer_auto_device_map (with CPU budget):')
for k, v_map in device_map.items():
    print(f'  {k}: {v_map}')
same_device_modules = ['language_model.model.embed_tokens', 'time_embedder',
    'latent_pos_embed', 'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed']
if torch.cuda.device_count() > 0:
    first_device_key = same_device_modules[0]
    default_target_device = 'cuda:0'
    first_module_target_device = device_map.get(first_device_key,
        default_target_device)
    print(
        f'Target device for same_device_modules (based on {first_device_key}): {first_module_target_device}'
        )
    for k_module in same_device_modules:
        if k_module in device_map:
            if device_map[k_module] != first_module_target_device:
                print(
                    f'  Moving {k_module} from {device_map[k_module]} to {first_module_target_device} (same_device_modules)'
                    )
                device_map[k_module] = first_module_target_device
        else:
            print(
                f'  Assigning {k_module} (from same_device_modules) to {first_module_target_device} as it was not in initial map.'
                )
            device_map[k_module] = first_module_target_device
elif torch.cuda.device_count() == 0 and 'cpu' in max_memory_config:
    print('No CUDA devices found. Assigning same_device_modules to CPU.')
    for k_module in same_device_modules:
        device_map[k_module] = 'cpu'
print('Device map after same_device_modules logic:')
for k, v_map in device_map.items():
    print(f'  {k}: {v_map}')
keys_to_change_to_cpu = []
for module_name, device_assignment in device_map.items():
    if device_assignment == 'disk':
        keys_to_change_to_cpu.append(module_name)
if keys_to_change_to_cpu:
    print(
        f"Manually changing the following layers from 'disk' to 'cpu': {keys_to_change_to_cpu}"
        )
    for module_name in keys_to_change_to_cpu:
        device_map[module_name] = 'cpu'
    print('Final device_map before loading checkpoint (after disk override):')
    for k, v_map in device_map.items():
        print(f'  {k}: {v_map}')
else:
    print(
        "No layers assigned to 'disk' by infer_auto_device_map, or they were already handled. Final device_map is as above."
        )
print(
    """
Starting custom device_map modifications to maximize GPU utilization..."""
    )
print('Device map state BEFORE custom modifications:')
for k_map_item, v_map_item in device_map.items():
    print(f'  {k_map_item}: {v_map_item}')
NUM_ADDITIONAL_LLM_LAYERS_TO_GPU = 5
TRY_MOVE_LLM_NORM_HEAD_TO_GPU = True
TRY_MOVE_VIT_MODEL_TO_GPU = False
moved_llm_layers_count = 0
if NUM_ADDITIONAL_LLM_LAYERS_TO_GPU > 0:
    print(
        f"""
Attempting to move up to {NUM_ADDITIONAL_LLM_LAYERS_TO_GPU} LLM layers (11 to {10 + NUM_ADDITIONAL_LLM_LAYERS_TO_GPU}) to GPU 0..."""
        )
    for i in range(NUM_ADDITIONAL_LLM_LAYERS_TO_GPU):
        layer_idx = 11 + i
        if layer_idx > 27:
            print(
                f'  Reached max layer index (27). Stopped LLM layer promotion.'
                )
            break
        layer_name = f'language_model.model.layers.{layer_idx}'
        if device_map.get(layer_name) == 'cpu':
            print(f"  Promoting LLM layer '{layer_name}' from 'cpu' to GPU 0.")
            device_map[layer_name] = 0
            moved_llm_layers_count += 1
        elif layer_name in device_map:
            print(
                f"  LLM Layer '{layer_name}' is already on device '{device_map[layer_name]}'. Skipping promotion."
                )
        else:
            print(
                f"  Warning: LLM Layer '{layer_name}' not found in device_map. Cannot promote."
                )
    print(
        f'Successfully promoted {moved_llm_layers_count} LLM layers to GPU 0.')
else:
    print(
        """
Skipping promotion of additional LLM layers based on NUM_ADDITIONAL_LLM_LAYERS_TO_GPU setting."""
        )
if TRY_MOVE_LLM_NORM_HEAD_TO_GPU:
    print(
        "\nAttempting to move LLM 'norm' and 'lm_head' to GPU 0 (if on CPU)..."
        )
    llm_aux_modules = ['language_model.model.norm',
        'language_model.model.lm_head']
    for module_name in llm_aux_modules:
        if device_map.get(module_name) == 'cpu':
            print(f"  Promoting '{module_name}' from 'cpu' to GPU 0.")
            device_map[module_name] = 0
        elif module_name in device_map:
            print(
                f"  Module '{module_name}' is already on device '{device_map[module_name]}'. Skipping promotion."
                )
        else:
            print(
                f"  Warning: Module '{module_name}' not found in device_map. Cannot promote."
                )
else:
    print(
        """
Skipping promotion of LLM 'norm' and 'lm_head' based on TRY_MOVE_LLM_NORM_HEAD_TO_GPU setting."""
        )
if TRY_MOVE_VIT_MODEL_TO_GPU:
    print("\nAttempting to move 'vit_model' to GPU 0 (if on CPU)...")
    vit_module_name = 'vit_model'
    if device_map.get(vit_module_name) == 'cpu':
        print(f"  Promoting '{vit_module_name}' from 'cpu' to GPU 0.")
        device_map[vit_module_name] = 0
    elif vit_module_name in device_map:
        print(
            f"  Module '{vit_module_name}' is already on device '{device_map[vit_module_name]}'. Skipping promotion."
            )
    else:
        print(
            f"  Warning: Module '{vit_module_name}' not found in device_map. Cannot promote."
            )
else:
    print(
        "\nSkipping promotion of 'vit_model' based on TRY_MOVE_VIT_MODEL_TO_GPU setting."
        )
print("""
Final device_map after all custom modifications:""")
for k_map_item, v_map_item in device_map.items():
    print(f'  {k_map_item}: {v_map_item}')
print('--- End of custom device_map modifications ---')
model = load_checkpoint_and_dispatch(model, checkpoint=checkpoint,
    device_map=device_map, offload_buffers=True, offload_folder='offload',
    dtype=torch.bfloat16, force_hooks=True).eval()
inferencer = InterleaveInferencer(model=model, vae_model=vae_model,
    tokenizer=tokenizer, vae_transform=vae_transform, vit_transform=
    vit_transform, new_token_ids=new_token_ids)
def set_seed(seed):
    """Set random seeds for reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
def text_to_image(prompt, show_thinking=False, cfg_text_scale=4.0,
    cfg_interval=0.4, timestep_shift=3.0, num_timesteps=50, cfg_renorm_min=
    1.0, cfg_renorm_type='global', max_think_token_n=1024, do_sample=False,
    text_temperature=0.3, seed=0, image_ratio='1:1'):
    set_seed(seed)
    if image_ratio == '1:1':
        image_shapes = 1024, 1024
    elif image_ratio == '4:3':
        image_shapes = 768, 1024
    elif image_ratio == '3:4':
        image_shapes = 1024, 768
    elif image_ratio == '16:9':
        image_shapes = 576, 1024
    elif image_ratio == '9:16':
        image_shapes = 1024, 576
    inference_hyper = dict(max_think_token_n=max_think_token_n if
        show_thinking else 1024, do_sample=do_sample if show_thinking else 
        False, text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale, cfg_interval=[cfg_interval, 1.0],
        timestep_shift=timestep_shift, num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min, cfg_renorm_type=cfg_renorm_type,
        image_shapes=image_shapes)
    start_time = time.time()
    result = inferencer(text=prompt, think=show_thinking, **inference_hyper)
    end_time = time.time()
    duration = end_time - start_time
    duration_str = f'{duration:.2f} seconds'
    print(f'The image takes time: {duration_str}')
    return result['image'], result.get('text', None), duration_str
def image_understanding(image, prompt, show_thinking=False, do_sample=False,
    text_temperature=0.3, max_new_tokens=512):
    start_time = time.time()
    if image is None:
        return 'Please upload an image.'
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = pil_img2rgb(image)
    inference_hyper = dict(do_sample=do_sample, text_temperature=
        text_temperature, max_think_token_n=max_new_tokens)
    result = inferencer(image=image, text=prompt, think=show_thinking,
        understanding_output=True, **inference_hyper)
    end_time = time.time()
    duration = end_time - start_time
    duration_str = f'{duration:.2f} seconds'
    print(f'The image takes time: {duration_str}')
    return result['text'], duration_str
def edit_image(image, prompt, show_thinking=False, cfg_text_scale=4.0,
    cfg_img_scale=2.0, cfg_interval=0.0, timestep_shift=3.0, num_timesteps=
    50, cfg_renorm_min=1.0, cfg_renorm_type='text_channel',
    max_think_token_n=1024, do_sample=False, text_temperature=0.3, seed=0):
    set_seed(seed)
    start_time = time.time()
    if image is None:
        return 'Please upload an image.', ''
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = pil_img2rgb(image)
    inference_hyper = dict(max_think_token_n=max_think_token_n if
        show_thinking else 1024, do_sample=do_sample if show_thinking else 
        False, text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale, cfg_img_scale=cfg_img_scale,
        cfg_interval=[cfg_interval, 1.0], timestep_shift=timestep_shift,
        num_timesteps=num_timesteps, cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type)
    result = inferencer(image=image, text=prompt, think=show_thinking, **
        inference_hyper)
    end_time = time.time()
    duration = end_time - start_time
    duration_str = f'{duration:.2f} seconds'
    print(f'The image takes time: {duration_str}')
    return result['image'], result.get('text', ''), duration_str
def load_example_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f'Error loading example image: {e}')
        return None
block = gr.Blocks(title='gguf').queue()
with block:
    with gr.Tab('📝 Text to Image'):
        txt_input = gr.Textbox(label='Prompt', value=
            'A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere.'
            )
        with gr.Row():
            show_thinking = gr.Checkbox(label='Thinking', value=False)
        with gr.Accordion('Inference Hyperparameters', open=False):
            with gr.Group():
                with gr.Row():
                    seed = gr.Slider(minimum=0, maximum=1000000, value=0,
                        step=1, label='Seed', info=
                        '0 for random seed, positive for reproducible results')
                    image_ratio = gr.Dropdown(choices=['1:1', '4:3', '3:4',
                        '16:9', '9:16'], value='1:1', label='Image Ratio',
                        info='The longer size is fixed to 1024')
                with gr.Row():
                    cfg_text_scale = gr.Slider(minimum=1.0, maximum=8.0,
                        value=4.0, step=0.1, interactive=True, label=
                        'CFG Text Scale', info=
                        'Controls how strongly the model follows the text prompt (4.0-8.0)'
                        )
                    cfg_interval = gr.Slider(minimum=0.0, maximum=1.0,
                        value=0.4, step=0.1, label='CFG Interval', info=
                        'Start of CFG application interval (end is fixed at 1.0)'
                        )
                with gr.Row():
                    cfg_renorm_type = gr.Dropdown(choices=['global',
                        'local', 'text_channel'], value='global', label=
                        'CFG Renorm Type', info=
                        "If the genrated image is blurry, use 'global'")
                    cfg_renorm_min = gr.Slider(minimum=0.0, maximum=1.0,
                        value=0.0, step=0.1, interactive=True, label=
                        'CFG Renorm Min', info='1.0 disables CFG-Renorm')
                with gr.Row():
                    num_timesteps = gr.Slider(minimum=10, maximum=100,
                        value=50, step=5, interactive=True, label=
                        'Timesteps', info='Total denoising steps')
                    timestep_shift = gr.Slider(minimum=1.0, maximum=5.0,
                        value=3.0, step=0.5, interactive=True, label=
                        'Timestep Shift', info=
                        'Higher values for layout, lower for details')
                thinking_params = gr.Group(visible=False)
                with thinking_params:
                    with gr.Row():
                        do_sample = gr.Checkbox(label='Sampling', value=
                            False, info='Enable sampling for text generation')
                        max_think_token_n = gr.Slider(minimum=64, maximum=
                            4006, value=1024, step=64, interactive=True,
                            label='Max Think Tokens', info=
                            'Maximum number of tokens for thinking')
                        text_temperature = gr.Slider(minimum=0.1, maximum=
                            1.0, value=0.3, step=0.1, interactive=True,
                            label='Temperature', info=
                            'Controls randomness in text generation')
        thinking_output = gr.Textbox(label='Thinking Process', visible=False)
        img_output = gr.Image(label='Generated Image')
        gen_btn = gr.Button('Generate', variant='primary')
        generation_time_output = gr.Textbox(label='Processing Time',
            interactive=False)
        def update_thinking_visibility(show):
            return gr.update(visible=show), gr.update(visible=show)
        show_thinking.change(fn=update_thinking_visibility, inputs=[
            show_thinking], outputs=[thinking_output, thinking_params])
        def process_text_to_image(prompt, show_thinking, cfg_text_scale,
            cfg_interval, timestep_shift, num_timesteps, cfg_renorm_min,
            cfg_renorm_type, max_think_token_n, do_sample, text_temperature,
            seed, image_ratio):
            image, thinking, duration_str = text_to_image(prompt,
                show_thinking, cfg_text_scale, cfg_interval, timestep_shift,
                num_timesteps, cfg_renorm_min, cfg_renorm_type,
                max_think_token_n, do_sample, text_temperature, seed,
                image_ratio)
            return image, thinking if thinking else '', duration_str
        gr.on(triggers=[gen_btn.click, txt_input.submit], fn=
            process_text_to_image, inputs=[txt_input, show_thinking,
            cfg_text_scale, cfg_interval, timestep_shift, num_timesteps,
            cfg_renorm_min, cfg_renorm_type, max_think_token_n, do_sample,
            text_temperature, seed, image_ratio], outputs=[img_output,
            thinking_output, generation_time_output])
    with gr.Tab('🖌️ Image Edit'):
        with gr.Row():
            with gr.Column(scale=1):
                edit_image_input = gr.Image(label='Input Image', value=
                    load_example_image(os.path.join(os.path.dirname(
                    __file__), '../bagel2/test_images/women.jpg')))
                edit_prompt = gr.Textbox(label='Prompt', value=
                    'She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes.'
                    )
            with gr.Column(scale=1):
                edit_image_output = gr.Image(label='Result')
                edit_thinking_output = gr.Textbox(label='Thinking Process',
                    visible=False)
        with gr.Row():
            edit_show_thinking = gr.Checkbox(label='Thinking', value=False)
        with gr.Accordion('Inference Hyperparameters', open=False):
            with gr.Group():
                with gr.Row():
                    edit_seed = gr.Slider(minimum=0, maximum=1000000, value
                        =0, step=1, interactive=True, label='Seed', info=
                        '0 for random seed, positive for reproducible results')
                    edit_cfg_text_scale = gr.Slider(minimum=1.0, maximum=
                        8.0, value=4.0, step=0.1, interactive=True, label=
                        'CFG Text Scale', info=
                        'Controls how strongly the model follows the text prompt'
                        )
                with gr.Row():
                    edit_cfg_img_scale = gr.Slider(minimum=1.0, maximum=4.0,
                        value=2.0, step=0.1, interactive=True, label=
                        'CFG Image Scale', info=
                        'Controls how much the model preserves input image details'
                        )
                    edit_cfg_interval = gr.Slider(minimum=0.0, maximum=1.0,
                        value=0.0, step=0.1, interactive=True, label=
                        'CFG Interval', info=
                        'Start of CFG application interval (end is fixed at 1.0)'
                        )
                with gr.Row():
                    edit_cfg_renorm_type = gr.Dropdown(choices=['global',
                        'local', 'text_channel'], value='text_channel',
                        label='CFG Renorm Type', info=
                        "If the genrated image is blurry, use 'global")
                    edit_cfg_renorm_min = gr.Slider(minimum=0.0, maximum=
                        1.0, value=0.0, step=0.1, interactive=True, label=
                        'CFG Renorm Min', info='1.0 disables CFG-Renorm')
                with gr.Row():
                    edit_num_timesteps = gr.Slider(minimum=10, maximum=100,
                        value=50, step=5, interactive=True, label=
                        'Timesteps', info='Total denoising steps')
                    edit_timestep_shift = gr.Slider(minimum=1.0, maximum=
                        10.0, value=3.0, step=0.5, interactive=True, label=
                        'Timestep Shift', info=
                        'Higher values for layout, lower for details')
                edit_thinking_params = gr.Group(visible=False)
                with edit_thinking_params:
                    with gr.Row():
                        edit_do_sample = gr.Checkbox(label='Sampling',
                            value=False, info=
                            'Enable sampling for text generation')
                        edit_max_think_token_n = gr.Slider(minimum=64,
                            maximum=4006, value=1024, step=64, interactive=
                            True, label='Max Think Tokens', info=
                            'Maximum number of tokens for thinking')
                        edit_text_temperature = gr.Slider(minimum=0.1,
                            maximum=1.0, value=0.3, step=0.1, interactive=
                            True, label='Temperature', info=
                            'Controls randomness in text generation')
        edit_btn = gr.Button('Submit', variant='primary')
        edit_generation_time_output = gr.Textbox(label='Processing Time',
            interactive=False)
        def update_edit_thinking_visibility(show):
            return gr.update(visible=show), gr.update(visible=show)
        edit_show_thinking.change(fn=update_edit_thinking_visibility,
            inputs=[edit_show_thinking], outputs=[edit_thinking_output,
            edit_thinking_params])
        def process_edit_image(image, prompt, show_thinking, cfg_text_scale,
            cfg_img_scale, cfg_interval, timestep_shift, num_timesteps,
            cfg_renorm_min, cfg_renorm_type, max_think_token_n, do_sample,
            text_temperature, seed):
            edited_image, thinking, duration_str = edit_image(image, prompt,
                show_thinking, cfg_text_scale, cfg_img_scale, cfg_interval,
                timestep_shift, num_timesteps, cfg_renorm_min,
                cfg_renorm_type, max_think_token_n, do_sample,
                text_temperature, seed)
            return edited_image, thinking if thinking else '', duration_str
        gr.on(triggers=[edit_btn.click, edit_prompt.submit], fn=
            process_edit_image, inputs=[edit_image_input, edit_prompt,
            edit_show_thinking, edit_cfg_text_scale, edit_cfg_img_scale,
            edit_cfg_interval, edit_timestep_shift, edit_num_timesteps,
            edit_cfg_renorm_min, edit_cfg_renorm_type,
            edit_max_think_token_n, edit_do_sample, edit_text_temperature,
            edit_seed], outputs=[edit_image_output, edit_thinking_output,
            edit_generation_time_output])
    with gr.Tab('🖼️ Image Understanding'):
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(label='Input Image', value=
                    load_example_image(os.path.join(os.path.dirname(
                    __file__), '../bagel2/test_images/meme.jpg')))
                understand_prompt = gr.Textbox(label='Prompt', value=
                    "Can someone explain what's funny about this meme??")
            with gr.Column(scale=1):
                txt_output = gr.Textbox(label='Result', lines=20)
        with gr.Row():
            understand_show_thinking = gr.Checkbox(label='Thinking', value=
                False)
        with gr.Accordion('Inference Hyperparameters', open=False):
            with gr.Row():
                understand_do_sample = gr.Checkbox(label='Sampling', value=
                    False, info='Enable sampling for text generation')
                understand_text_temperature = gr.Slider(minimum=0.0,
                    maximum=1.0, value=0.3, step=0.05, interactive=True,
                    label='Temperature', info=
                    'Controls randomness in text generation (0=deterministic, 1=creative)'
                    )
                understand_max_new_tokens = gr.Slider(minimum=64, maximum=
                    4096, value=512, step=64, interactive=True, label=
                    'Max New Tokens', info=
                    'Maximum length of generated text, including potential thinking'
                    )
        img_understand_btn = gr.Button('Submit', variant='primary')
        understand_generation_time_output = gr.Textbox(label=
            'Processing Time', interactive=False)
        def process_understanding(image, prompt, show_thinking, do_sample,
            text_temperature, max_new_tokens):
            result, duration_str = image_understanding(image, prompt,
                show_thinking, do_sample, text_temperature, max_new_tokens)
            return result, duration_str
        gr.on(triggers=[img_understand_btn.click, understand_prompt.submit],
            fn=process_understanding, inputs=[img_input, understand_prompt,
            understand_show_thinking, understand_do_sample,
            understand_text_temperature, understand_max_new_tokens],
            outputs=[txt_output, understand_generation_time_output])
    with gr.Tab('📊 System Monitor'):
        with gr.Column():
            memory_stats_display = gr.Markdown('Check RAM/VRAM Stats')
            refresh_button = gr.Button('🔄 Check RAM/VRAM Stats')
            refresh_button.click(fn=get_all_memory_stats_for_gradio_display,
                inputs=None, outputs=[memory_stats_display])
block.launch()