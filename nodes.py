import nodes
import torch
import comfy.model_management as mm
from PIL import ImageDraw, Image, ImageChops, ImageColor
import numpy as np
from typing import List, Set, Union, Optional, Dict, Optional, Tuple
import struct
import folder_paths
from pathlib import Path
import yaml

from comfy import utils as utils
from comfy import samplers as samplers
from comfy import sampler_helpers as sampler_helpers
from comfy import patcher_extension as patcher_extension
from comfy import model_patcher as comfy_model_patcher
from comfy import hooks as hooks
from comfy import model_sampling as model_sampling

from comfy.samplers import preprocess_conds_hooks as preprocess_conds_hooks
from comfy.samplers import get_total_hook_groups_in_conds as get_total_hook_groups_in_conds
from comfy.samplers import filter_registered_hooks_on_conds as filter_registered_hooks_on_conds
from comfy.samplers import sampling_function as sampling_function
from comfy.samplers import cast_to_load_options as cast_to_load_options
from comfy.samplers import process_conds as process_conds

import os
import gc
import time
import json
import latent_preview
from decimal import *
import traceback
import sys
import cv2
import glob
import comfy
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from comfy.cli_args import args
from PIL.PngImagePlugin import PngInfo
import random
from tqdm import tqdm

import node_helpers

from torch import Tensor
from einops import repeat
from unittest.mock import patch

from comfy.ldm.flux.layers import timestep_embedding, apply_mod
from comfy.ldm.lightricks.model import precompute_freqs_cis
from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
from comfy.ldm.common_dit import rms_norm
from comfy.ldm.wan.model import sinusoidal_embedding_1d

from .diffusers_helper.memory import move_model_to_device_with_memory_preservation
from .diffusers_helper.k_diffusion_hunyuan import sample_hunyuan, sample_hunyuan2
from .diffusers_helper.utils import crop_or_pad_yield_mask, soft_append_bcthw
import math

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_VIDEO_EXT = ('mp4', 'flv', 'mov', 'avi', 'mpg', 'webm', 'mkv')

def get_offload_device():
    return torch.device("cpu")

def tensor2pil(image):
    # _tensor_check_image(image)
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))

def pil2tensor(image, device=None):
    if device is None:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).to(device).unsqueeze(0)

def pil2tensorSwap(image: Union[Image.Image, List[Image.Image]], device=None) -> torch.Tensor:
    if isinstance(image, list):
        if device is None:
            return torch.cat([pil2tensor(img) for img in image], dim=0)

        new_tensor = torch.cat([pil2tensor(img, device=device) for img in image], dim=0)
        # new_tensor = new_tensor.to(get_offload_device())

        return new_tensor

    if device is None:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    new_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).to(device).unsqueeze(0)
    # new_tensor = new_tensor.to(get_offload_device())

    return new_tensor

def tensor2pilSwap(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]

def get_lora_list():
    templist = folder_paths.get_filename_list("loras")

    i = 0
    while i < len(templist):
        templist[i] = templist[i].lower()
        i += 1

    return templist

def get_default_output_path():
    default_path = "{}\\MergedHunyuanLoras\\new_lora_hy.safetensors".format(folder_paths.output_directory)
    return default_path

def get_default_output_folder():
    default_folder = "{}\\MergedHunyuanLoras".format(folder_paths.output_directory)
    return default_folder

def get_default_wan_output_path():
    default_path = "{}\\MergedWanLoras\\new_lora_wan.safetensors".format(folder_paths.output_directory)
    return default_path

def get_default_wan_output_folder():
    default_folder = "{}\\MergedWanLoras".format(folder_paths.output_directory)
    return default_folder

SUPPORTED_MODELS_COEFFICIENTS = {
    "hunyuan_video": [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02],
    "wan2.1_t2v_1.3B": [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
    "wan2.1_t2v_14B": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
    "wan2.1_i2v_480p_14B": [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01],
    "wan2.1_i2v_720p_14B": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683]
}

INITIAL_COEFFICIENTS = [1.0, 1.0, 1.0, 1.0, 1.0]

def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result

def load_upscale_model(model_name):
    model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
    print("Upscale Model Path for {}: {}".format(model_name, model_path))
    sd = utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = utils.state_dict_prefix_replace(sd, {"module.":""})
    out = ModelLoader().load_from_state_dict(sd).eval()

    if not isinstance(out, ImageModelDescriptor):
        raise Exception("Upscale model must be a single-image model.")

    return out

def upscale_with_model(upscale_model, images, upscale_by, rescale_method):
    __imageScaler = ImageUpscaleWithModel()
    upscaled_images = []

    samples = images.movedim(-1,1)

    width = round(samples.shape[3])
    height = round(samples.shape[2])

    target_width = round(samples.shape[3] * upscale_by)
    target_height = round(samples.shape[2] * upscale_by)

    samples = __imageScaler.upscale(upscale_model, images)[0].movedim(-1,1)

    upscaled_width = round(samples.shape[3])
    upscaled_height = round(samples.shape[2])

    if upscaled_width > target_width or upscaled_height > target_height:
        samples = utils.common_upscale(samples, target_width, target_height, rescale_method, "disabled")

    samples = samples.movedim(1,-1)

    return samples

def upscale_latents_by(latent, upscale_method, scale_by):
    s = latent.clone()
    width = round(latent.shape[-1] * scale_by)
    height = round(latent.shape[-2] * scale_by)

    s = utils.common_upscale(latent, width, height, upscale_method, "disabled")
    return s

def partial_encode_basic(vae, image):
    latents = vae.encode(image[:,:,:,:3])
    return latents

def partial_encode_tiled(vae, image, tile_size=256, overlap=64, temporal_size=64, temporal_overlap=8, use_type=torch.float32, unload_after=True):
    latents = vae.encode_tiled(image[:,:,:,:3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
    latents = latents.to(dtype=use_type, device=get_offload_device())

    if unload_after:
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

    return latents

def partial_decode_basic(vae, latents):
    images = vae.decode(latents)
    if len(images.shape) == 5: #Combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images

def partial_decode_tiled(vae, latents, tile_size, overlap=64, temporal_size=64, temporal_overlap=8, unload_after=True):
    if tile_size < overlap * 4:
        overlap = tile_size // 4
    if temporal_size < temporal_overlap * 2:
        temporal_overlap = temporal_overlap // 2
    temporal_compression = vae.temporal_compression_decode()
    if temporal_compression is not None:
        temporal_size = max(2, temporal_size // temporal_compression)
        temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
    else:
        temporal_size = None
        temporal_overlap = None

    compression = vae.spacial_compression_decode()
    images = vae.decode_tiled(latents, tile_x=tile_size // compression, tile_y=tile_size // compression, overlap=overlap // compression, tile_t=temporal_size, overlap_t=temporal_overlap)
    if len(images.shape) == 5: #Combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

    if unload_after:
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

    return images

class HyVideoModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}
        self.load_device = mm.get_torch_device()

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v


class HyVideoModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = comfy.latent_formats.HunyuanVideo
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        self.memory_usage_factor = 2.0
        self.unet_config["disable_unet_model_creation"] = True

# def partial_decode_tiled(vae, latents, tile_size, overlap=64, temporal_size=64, temporal_overlap=8, unload_after=True):
#     if tile_size < overlap * 4:
#         overlap = tile_size // 4
#     if temporal_size < temporal_overlap * 2:
#         temporal_overlap = temporal_overlap // 2
#     temporal_compression = vae.temporal_compression_decode()
#     if temporal_compression is not None:
#         temporal_size = max(2, temporal_size // temporal_compression)
#         temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
#     else:
#         temporal_size = None
#         temporal_overlap = None
#
#     compression = vae.spacial_compression_decode()
#     images = vae.decode_tiled(latents, tile_x=tile_size // compression, tile_y=tile_size // compression, overlap=overlap // compression, tile_t=temporal_size, overlap_t=temporal_overlap)
#     if len(images.shape) == 5: #Combine batches
#         images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
#
#     if unload_after:
#         mm.unload_all_models()
#         mm.soft_empty_cache()
#         gc.collect()
#         time.sleep(1)
#
#     return images

def warped_prepare_noise(latent_image, seed, generator=None):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    if generator is None:
        generator = torch.manual_seed(seed)

    output_result = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")

    return output_result

def warped_prepare_noise_images(images, seed, generator=None):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    if generator is None:
        generator = torch.manual_seed(seed)

    output_result = torch.randn(images.size(), dtype=images.dtype, layout=images.layout, generator=generator, device="cpu")

    return output_result

def get_upscale_methods():
    return ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

def get_rescale_methods():
    return ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

class WarpedHunyuanLoraMerge:
    def __init__(self):
        self.base_output_dir = get_default_output_folder()
        os.makedirs(self.base_output_dir, exist_ok = True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_path": ("STRING", {"default": get_default_output_path()}),
                "lora_1": (['None'] + get_lora_list(),),
                "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "blocks_type_1": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_2": (['None'] + get_lora_list(),),
                "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "blocks_type_2": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "merge_multiple_loras"
    CATEGORY = "Warped/HunyuanTools"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

    def convert_key_format(self, key: str) -> str:
        """Standardize LoRA key format by removing prefixes."""
        prefixes = ["diffusion_model.", "transformer."]
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        return key

    def filter_lora_keys(self, lora: Dict[str, torch.Tensor], blocks_type: str) -> Dict[str, torch.Tensor]:
        """Filter LoRA weights based on block type."""
        if blocks_type == "all":
            return lora
        filtered_lora = {}
        for key, value in lora.items():
            base_key = self.convert_key_format(key)
            if blocks_type in base_key:
                filtered_lora[key] = value
        return filtered_lora

    def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load and filter a single LoRA model."""
        if not lora_name or strength == 0:
            return {}, {}

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        # Load the LoRA weights
        lora_weights = utils.load_torch_file(lora_path)

        # Filter the LoRA weights based on the block type
        filtered_lora = self.filter_lora_keys(lora_weights, blocks_type)

        return lora_weights, filtered_lora

    def merge_multiple_loras(self, save_path, lora_1, strength_1, blocks_type_1, lora_2, strength_2, blocks_type_2, save_metadata=True):
        """Load and apply multiple LoRA models."""
        temp_loras = {}
        metadata = {"loras": "{} and {}".format(lora_1, lora_2)}
        metadata["strengths"] = "{} and {}".format(strength_1, strength_2)
        metadata["block_types"] = "{} and {}".format(blocks_type_1, blocks_type_2)

        if lora_1 != "None" and strength_1 != 0:
            # Load and filter the LoRA weights
            lora_weights, filtered_lora = self.load_lora(lora_1, 1.0, blocks_type_1)
            temp_loras["1"] = {"lora_weights": lora_weights, "strength": strength_1, "filtered_lora": filtered_lora}

        if lora_2 != "None" and strength_2 != 0:
            # Load and filter the LoRA weights
            lora_weights, filtered_lora = self.load_lora(lora_2, 1.0, blocks_type_2)
            temp_loras["2"] = {"lora_weights": lora_weights, "strength": strength_2, "filtered_lora": filtered_lora}

        loras = {}

        for lora_key in temp_loras.keys():
            loras[lora_key] = {"lora_weights": {}, "strength": temp_loras[lora_key]["strength"], "filtered_lora": temp_loras[lora_key]["filtered_lora"]}

            for key in temp_loras[lora_key]["lora_weights"].keys():
                new_key = key.replace("transformer.", "diffusion_model.")
                loras[lora_key]["lora_weights"][new_key] = temp_loras[lora_key]["lora_weights"][key]

        new_lora = {}

        for lora_key in loras.keys():
            for key in loras[lora_key]["lora_weights"].keys():
                if not key in new_lora.keys():
                    new_lora[key] = None
                print("Lora: {}  | Key: {}  |  Shape: {}".format(lora_key, key, loras[lora_key]["lora_weights"][key].shape))

        # print_it = True
        #
        # i = 0

        # Merge The Weighted Key Weights
        for key in new_lora.keys():
            for lora_key in loras.keys():
                if key in loras[lora_key]["lora_weights"].keys():
                    if not new_lora[key] is None:
                        temp_weights = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"])

                        # if print_it:
                        #     print("\n--------------------------------------------------------------------")
                        #     print("Before: New Lora Shape: {}  |  Temp Weights Shape: {}".format(new_lora[key].shape, temp_weights.shape))
                        #     print("--------------------------------------------------------------------\n")

                        if new_lora[key].shape[0] < new_lora[key].shape[1]:
                            if temp_weights.shape[0] < new_lora[key].shape[0]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([new_lora[key].shape[0], new_lora[key].shape[1]])
                                padding[:temp_weights.shape[0],:] = temp_weights
                                temp_weights = padding
                            elif temp_weights.shape[0] > new_lora[key].shape[0]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([temp_weights.shape[0], temp_weights.shape[1]])
                                padding[:new_lora[key].shape[0],:] = new_lora[key]
                                new_lora[key] = padding
                        else:
                            if temp_weights.shape[1] < new_lora[key].shape[1]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([new_lora[key].shape[0], new_lora[key].shape[1]])
                                padding[:,:temp_weights.shape[1]] = temp_weights
                                temp_weights = padding
                            elif temp_weights.shape[1] > new_lora[key].shape[1]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([temp_weights.shape[0], temp_weights.shape[1]])
                                padding[:,:new_lora[key].shape[1]] = new_lora[key]
                                new_lora[key] = padding

                        try:
                            # new_lora[key] = ((new_lora[key] + temp_weights) / 2.0)
                            new_lora[key] = torch.add(new_lora[key], temp_weights)

                            # if print_it:
                            #     if (i + 1) >= 10:
                            #         print_it = False
                            #
                            #     i += 1
                            #     print("\n--------------------------------------------------------------------")
                            #     print("After: New Lora Shape: {}".format(new_lora[key].shape))
                            #     print("--------------------------------------------------------------------\n")
                                # print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                                # print("After: New Lora Key Tensor: {}".format(new_lora[key]))
                                # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                        except Exception as e:
                            raise(e)
                    else:
                        new_lora[key] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"])

        if not save_metadata:
            metadata = None

        utils.save_torch_file(new_lora, save_path, metadata=metadata)

        save_message = "Weights Saved To: {}".format(save_path)

        return {"ui": {"tags": [save_message]}}

class WarpedHunyuanMultiLoraMerge:
    def __init__(self):
        self.base_output_dir = get_default_output_folder()
        os.makedirs(self.base_output_dir, exist_ok = True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_path": ("STRING", {"default": get_default_output_path()}),
                "lora_1": (['None'] + get_lora_list(),),
                "strength_1": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "blocks_type_1": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_2": (['None'] + get_lora_list(),),
                "strength_2": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "blocks_type_2": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_3": (['None'] + get_lora_list(),),
                "strength_3": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "blocks_type_3": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_4": (['None'] + get_lora_list(),),
                "strength_4": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "blocks_type_4": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "merge_multiple_loras"
    CATEGORY = "Warped/HunyuanTools"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

    def convert_key_format(self, key: str) -> str:
        """Standardize LoRA key format by removing prefixes."""
        prefixes = ["diffusion_model.", "transformer."]
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        return key

    def filter_lora_keys(self, lora: Dict[str, torch.Tensor], blocks_type: str) -> Dict[str, torch.Tensor]:
        """Filter LoRA weights based on block type."""
        if blocks_type == "all":
            return lora
        filtered_lora = {}
        for key, value in lora.items():
            base_key = self.convert_key_format(key)
            if blocks_type in base_key:
                filtered_lora[key] = value
        return filtered_lora

    def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load and filter a single LoRA model."""
        if not lora_name or strength == 0:
            return {}, {}

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        # Load the LoRA weights
        lora_weights = utils.load_torch_file(lora_path)

        # Filter the LoRA weights based on the block type
        filtered_lora = self.filter_lora_keys(lora_weights, blocks_type)

        return lora_weights, filtered_lora

    def merge_multiple_loras(self, save_path, lora_1, strength_1, blocks_type_1, lora_2, strength_2, blocks_type_2, lora_3, strength_3, blocks_type_3, lora_4, strength_4, blocks_type_4, save_metadata=True):
        temp_loras = {}
        metadata = {"loras": "{} and {} and {} and {}".format(lora_1, lora_2, lora_3, lora_4)}
        metadata["strengths"] = "{} and {} and {} and {}".format(strength_1, strength_2, strength_3, strength_4)
        metadata["block_types"] = "{} and {} and {} and {}".format(blocks_type_1, blocks_type_2, blocks_type_3, blocks_type_4)

        if lora_1 != "None" and strength_1 != 0:
            # Load and filter the LoRA weights
            lora_weights, filtered_lora = self.load_lora(lora_1, 1.0, blocks_type_1)
            temp_loras["1"] = {"lora_weights": lora_weights, "strength": strength_1, "filtered_lora": filtered_lora}

        if lora_2 != "None" and strength_2 != 0:
            # Load and filter the LoRA weights
            lora_weights, filtered_lora = self.load_lora(lora_2, 1.0, blocks_type_2)
            temp_loras["2"] = {"lora_weights": lora_weights, "strength": strength_2, "filtered_lora": filtered_lora}

        if lora_3 != "None" and strength_3 != 0:
            # Load and filter the LoRA weights
            lora_weights, filtered_lora = self.load_lora(lora_3, 1.0, blocks_type_3)
            temp_loras["3"] = {"lora_weights": lora_weights, "strength": strength_3, "filtered_lora": filtered_lora}

        if lora_4 != "None" and strength_4 != 0:
            # Load and filter the LoRA weights
            lora_weights, filtered_lora = self.load_lora(lora_4, 1.0, blocks_type_4)
            temp_loras["4"] = {"lora_weights": lora_weights, "strength": strength_4, "filtered_lora": filtered_lora}

        loras = {}

        for lora_key in temp_loras.keys():
            loras[lora_key] = {"lora_weights": {}, "strength": temp_loras[lora_key]["strength"], "filtered_lora": temp_loras[lora_key]["filtered_lora"]}

            for key in temp_loras[lora_key]["lora_weights"].keys():
                new_key = key.replace("transformer.", "diffusion_model.")
                loras[lora_key]["lora_weights"][new_key] = temp_loras[lora_key]["lora_weights"][key]

        new_lora = {}

        for lora_key in loras.keys():
            for key in loras[lora_key]["lora_weights"].keys():
                if not key in new_lora.keys():
                    new_lora[key] = None
                print("Lora: {}  | Key: {}".format(lora_key, key))

        # print_it = True
        #
        # i = 0

        # Merge The Weighted Key Weights
        for key in new_lora.keys():
            for lora_key in loras.keys():
                if key in loras[lora_key]["lora_weights"].keys():
                    if not new_lora[key] is None:
                        temp_weights = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"])

                        # if print_it:
                        #     print("\n--------------------------------------------------------------------")
                        #     print("Before: New Lora Shape: {}  |  Temp Weights Shape: {}".format(new_lora[key].shape, temp_weights.shape))
                        #     print("--------------------------------------------------------------------\n")

                        if new_lora[key].shape[0] < new_lora[key].shape[1]:
                            if temp_weights.shape[0] < new_lora[key].shape[0]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([new_lora[key].shape[0], new_lora[key].shape[1]])
                                padding[:temp_weights.shape[0],:] = temp_weights
                                temp_weights = padding
                            elif temp_weights.shape[0] > new_lora[key].shape[0]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([temp_weights.shape[0], temp_weights.shape[1]])
                                padding[:new_lora[key].shape[0],:] = new_lora[key]
                                new_lora[key] = padding
                        else:
                            if temp_weights.shape[1] < new_lora[key].shape[1]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([new_lora[key].shape[0], new_lora[key].shape[1]])
                                padding[:,:temp_weights.shape[1]] = temp_weights
                                temp_weights = padding
                            elif temp_weights.shape[1] > new_lora[key].shape[1]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([temp_weights.shape[0], temp_weights.shape[1]])
                                padding[:,:new_lora[key].shape[1]] = new_lora[key]
                                new_lora[key] = padding

                        try:
                            new_lora[key] = torch.add(new_lora[key], temp_weights)
                        except Exception as e:
                            raise(e)
                    else:
                        new_lora[key] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"])

        if not save_metadata:
            metadata = None

        utils.save_torch_file(new_lora, save_path, metadata=metadata)

        save_message = "Weights Saved To: {}".format(save_path)

        return {"ui": {"tags": [save_message]}}

class WarpedHunyuanLoraAvgMerge:
    def __init__(self):
        self.base_output_dir = get_default_output_folder()
        os.makedirs(self.base_output_dir, exist_ok = True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_path": ("STRING", {"default": get_default_output_path()}),
                "lora_1": (['None'] + get_lora_list(),),
                # "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "blocks_type_1": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_2": (['None'] + get_lora_list(),),
                # "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "blocks_type_2": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "merge_multiple_loras"
    CATEGORY = "Warped/HunyuanTools"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

    def convert_key_format(self, key: str) -> str:
        """Standardize LoRA key format by removing prefixes."""
        prefixes = ["diffusion_model.", "transformer."]
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        return key

    def filter_lora_keys(self, lora: Dict[str, torch.Tensor], blocks_type: str) -> Dict[str, torch.Tensor]:
        """Filter LoRA weights based on block type."""
        if blocks_type == "all":
            return lora
        filtered_lora = {}
        for key, value in lora.items():
            base_key = self.convert_key_format(key)
            if blocks_type in base_key:
                filtered_lora[key] = value
        return filtered_lora

    def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load and filter a single LoRA model."""
        if not lora_name or strength == 0:
            return {}, {}

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        # Load the LoRA weights
        lora_weights = utils.load_torch_file(lora_path)

        # Filter the LoRA weights based on the block type
        filtered_lora = self.filter_lora_keys(lora_weights, blocks_type)

        return lora_weights, filtered_lora

    def merge_multiple_loras(self, save_path, lora_1, blocks_type_1, lora_2, blocks_type_2, save_metadata=True):
        """Load and apply multiple LoRA models."""
        strength = 1.0000
        temp_loras = {}
        metadata = {"loras": "{} and {}".format(lora_1, lora_2)}
        metadata["strengths"] = "{} and {}".format(strength, strength)
        metadata["block_types"] = "{} and {}".format(blocks_type_1, blocks_type_2)

        if lora_1 != "None":
            # Load and filter the LoRA weights
            lora_weights, filtered_lora = self.load_lora(lora_1, 1.0, blocks_type_1)
            temp_loras["1"] = {"lora_weights": lora_weights, "strength": strength, "filtered_lora": filtered_lora}

        if lora_2 != "None":
            # Load and filter the LoRA weights
            lora_weights, filtered_lora = self.load_lora(lora_2, 1.0, blocks_type_2)
            temp_loras["2"] = {"lora_weights": lora_weights, "strength": strength, "filtered_lora": filtered_lora}

        loras = {}

        for lora_key in temp_loras.keys():
            loras[lora_key] = {"lora_weights": {}, "strength": temp_loras[lora_key]["strength"], "filtered_lora": temp_loras[lora_key]["filtered_lora"]}

            for key in temp_loras[lora_key]["lora_weights"].keys():
                new_key = key.replace("transformer.", "diffusion_model.")
                loras[lora_key]["lora_weights"][new_key] = temp_loras[lora_key]["lora_weights"][key]

        new_lora = {}

        for lora_key in loras.keys():
            for key in loras[lora_key]["lora_weights"].keys():
                if not key in new_lora.keys():
                    new_lora[key] = None
                print("Lora: {}  | Key: {}".format(lora_key, key))

        # Merge The Weighted Key Weights
        for key in new_lora.keys():
            for lora_key in loras.keys():
                if key in loras[lora_key]["lora_weights"].keys():
                    if not new_lora[key] is None:
                        temp_weights = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"])

                        if new_lora[key].shape[0] < new_lora[key].shape[1]:
                            if temp_weights.shape[0] < new_lora[key].shape[0]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([new_lora[key].shape[0], new_lora[key].shape[1]])
                                padding[:temp_weights.shape[0],:] = temp_weights
                                temp_weights = padding
                            elif temp_weights.shape[0] > new_lora[key].shape[0]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([temp_weights.shape[0], temp_weights.shape[1]])
                                padding[:new_lora[key].shape[0],:] = new_lora[key]
                                new_lora[key] = padding
                        else:
                            if temp_weights.shape[1] < new_lora[key].shape[1]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([new_lora[key].shape[0], new_lora[key].shape[1]])
                                padding[:,:temp_weights.shape[1]] = temp_weights
                                temp_weights = padding
                            elif temp_weights.shape[1] > new_lora[key].shape[1]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([temp_weights.shape[0], temp_weights.shape[1]])
                                padding[:,:new_lora[key].shape[1]] = new_lora[key]
                                new_lora[key] = padding

                        try:
                            new_lora[key] = torch.add(new_lora[key], temp_weights)
                            new_lora[key] = torch.div(new_lora[key], 2.0000)
                        except Exception as e:
                            raise(e)
                    else:
                        new_lora[key] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"])

        if not save_metadata:
            metadata = None

        utils.save_torch_file(new_lora, save_path, metadata=metadata)

        save_message = "Weights Saved To: {}".format(save_path)

        return {"ui": {"tags": [save_message]}}

class WarpedHunyuanMultiLoraAvgMerge:
    def __init__(self):
        self.base_output_dir = get_default_output_folder()
        os.makedirs(self.base_output_dir, exist_ok = True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_path": ("STRING", {"default": get_default_output_path()}),
                "lora_1": (['None'] + get_lora_list(),),
                "blocks_type_1": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_2": (['None'] + get_lora_list(),),
                "blocks_type_2": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_3": (['None'] + get_lora_list(),),
                "blocks_type_3": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_4": (['None'] + get_lora_list(),),
                "blocks_type_4": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "merge_multiple_loras"
    CATEGORY = "Warped/HunyuanTools"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

    def convert_key_format(self, key: str) -> str:
        """Standardize LoRA key format by removing prefixes."""
        prefixes = ["diffusion_model.", "transformer."]
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        return key

    def filter_lora_keys(self, lora: Dict[str, torch.Tensor], blocks_type: str) -> Dict[str, torch.Tensor]:
        """Filter LoRA weights based on block type."""
        if blocks_type == "all":
            return lora
        filtered_lora = {}
        for key, value in lora.items():
            base_key = self.convert_key_format(key)
            if blocks_type in base_key:
                filtered_lora[key] = value
        return filtered_lora

    def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load and filter a single LoRA model."""
        if not lora_name or strength == 0:
            return {}, {}

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        # Load the LoRA weights
        lora_weights = utils.load_torch_file(lora_path)

        # Filter the LoRA weights based on the block type
        filtered_lora = self.filter_lora_keys(lora_weights, blocks_type)

        return lora_weights, filtered_lora

    def merge_multiple_loras(self, save_path, lora_1, blocks_type_1, lora_2, blocks_type_2, lora_3, blocks_type_3, lora_4, blocks_type_4, save_metadata=True):
        strength = 1.0000
        temp_loras = {}
        metadata = {"loras": "{} and {} and {} and {}".format(lora_1, lora_2, lora_3, lora_4)}
        metadata["strengths"] = "{} and {} and {} and {}".format(strength, strength, strength, strength)
        metadata["block_types"] = "{} and {} and {} and {}".format(blocks_type_1, blocks_type_2, blocks_type_3, blocks_type_4)

        if lora_1 != "None":
            # Load and filter the LoRA weights
            lora_weights, filtered_lora = self.load_lora(lora_1, 1.0, blocks_type_1)
            temp_loras["1"] = {"lora_weights": lora_weights, "strength": strength, "filtered_lora": filtered_lora}

        if lora_2 != "None":
            # Load and filter the LoRA weights
            lora_weights, filtered_lora = self.load_lora(lora_2, 1.0, blocks_type_2)
            temp_loras["2"] = {"lora_weights": lora_weights, "strength": strength, "filtered_lora": filtered_lora}

        if lora_3 != "None":
            # Load and filter the LoRA weights
            lora_weights, filtered_lora = self.load_lora(lora_3, 1.0, blocks_type_3)
            temp_loras["3"] = {"lora_weights": lora_weights, "strength": strength, "filtered_lora": filtered_lora}

        if lora_4 != "None":
            # Load and filter the LoRA weights
            lora_weights, filtered_lora = self.load_lora(lora_4, 1.0, blocks_type_4)
            temp_loras["4"] = {"lora_weights": lora_weights, "strength": strength, "filtered_lora": filtered_lora}

        loras = {}

        for lora_key in temp_loras.keys():
            loras[lora_key] = {"lora_weights": {}, "strength": temp_loras[lora_key]["strength"], "filtered_lora": temp_loras[lora_key]["filtered_lora"]}

            for key in temp_loras[lora_key]["lora_weights"].keys():
                new_key = key.replace("transformer.", "diffusion_model.")
                loras[lora_key]["lora_weights"][new_key] = temp_loras[lora_key]["lora_weights"][key]

        new_lora = {}
        num_loras = int(len(loras))

        for lora_key in loras.keys():
            for key in loras[lora_key]["lora_weights"].keys():
                if not key in new_lora.keys():
                    new_lora[key] = {"weights": None, "count": 0}
                print("Lora: {}  | Key: {}".format(lora_key, key))

        # Merge The Weighted Key Weights
        for key in new_lora.keys():
            for lora_key in loras.keys():
                if key in loras[lora_key]["lora_weights"].keys():
                    if not new_lora[key]["weights"] is None:
                        temp_weights = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"])

                        if new_lora[key]["weights"].shape[0] < new_lora[key]["weights"].shape[1]:
                            if temp_weights.shape[0] < new_lora[key]["weights"].shape[0]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key]["weights"] = new_lora[key]["weights"].clone().detach()

                                padding = torch.zeros([new_lora[key]["weights"].shape[0], new_lora[key]["weights"].shape[1]])
                                padding[:temp_weights.shape[0],:] = temp_weights
                                temp_weights = padding
                            elif temp_weights.shape[0] > new_lora[key]["weights"].shape[0]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key]["weights"] = new_lora[key]["weights"].clone().detach()

                                padding = torch.zeros([temp_weights.shape[0], temp_weights.shape[1]])
                                padding[:new_lora[key]["weights"].shape[0],:] = new_lora[key]["weights"]
                                new_lora[key]["weights"] = padding
                        else:
                            if temp_weights.shape[1] < new_lora[key]["weights"].shape[1]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key]["weights"] = new_lora[key]["weights"].clone().detach()

                                padding = torch.zeros([new_lora[key]["weights"].shape[0], new_lora[key]["weights"].shape[1]])
                                padding[:,:temp_weights.shape[1]] = temp_weights
                                temp_weights = padding
                            elif temp_weights.shape[1] > new_lora[key]["weights"].shape[1]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key]["weights"] = new_lora[key]["weights"].clone().detach()

                                padding = torch.zeros([temp_weights.shape[0], temp_weights.shape[1]])
                                padding[:,:new_lora[key]["weights"].shape[1]] = new_lora[key]["weights"]
                                new_lora[key]["weights"] = padding

                        try:
                            new_lora[key]["weights"] = torch.add(new_lora[key]["weights"], temp_weights)
                            new_lora[key]["count"] += 1
                        except Exception as e:
                            raise(e)
                    else:
                        new_lora[key]["weights"] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"])
                        new_lora[key]["count"] = 1

        final_lora = {}
        for key in new_lora.keys():
            final_lora[key] = torch.div(new_lora[key]["weights"], new_lora[key]["count"])

        if not save_metadata:
            metadata = None

        utils.save_torch_file(final_lora, save_path, metadata=metadata)

        save_message = "Weights Saved To: {}".format(save_path)

        return {"ui": {"tags": [save_message]}}

class WarpedHunyuanMultiLoraMixer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_folder": ("STRING", {"default": get_default_output_folder()}),
                "model_prefix": ("STRING", {"default": "new_model_hy"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "num_output": ("INT", {"default": 1, "min": 1, "max": 100}),
                "lora_1": (['None'] + get_lora_list(),),
                "lora_2": (['None'] + get_lora_list(),),
                "lora_3": (['None'] + get_lora_list(),),
                "lora_4": (['None'] + get_lora_list(),),
                "lora_5": (['None'] + get_lora_list(),),
                "lora_6": (['None'] + get_lora_list(),),
                "lora_7": (['None'] + get_lora_list(),),
                "lora_8": (['None'] + get_lora_list(),),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "merge_multiple_loras"
    CATEGORY = "Warped/HunyuanTools"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

    def convert_key_format(self, key: str) -> str:
        """Standardize LoRA key format by removing prefixes."""
        prefixes = ["diffusion_model.", "transformer."]
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        return key

    def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load and filter a single LoRA model."""
        if not lora_name or strength == 0:
            return {}, {}

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        # Load the LoRA weights
        lora_weights = utils.load_torch_file(lora_path)

        return lora_weights

    def get_mixtures(self, seed, num_output, lora_keys):
        random.seed(seed)
        mixtures = {}

        for i in range(num_output):
            mixtures["{}".format(i + 1)] = {}

        for key in lora_keys:
            for mixture_key in mixtures.keys():
                mixtures[mixture_key][key] = {"single": [], "double": []}

        for mixture_key in mixtures.keys():
            for j in range(40):
                temp_key = "{}".format(random.randint(1, len(lora_keys)))
                mixtures[mixture_key][temp_key]["single"].append(j)

            for j in range(20):
                temp_key = "{}".format(random.randint(1, len(lora_keys)))
                mixtures[mixture_key][temp_key]["double"].append(j)

            i += 1

        print("\nMixtures\n")

        block_metadata = ""

        for mixture_key in mixtures.keys():
            for key in mixtures[mixture_key]:
                print("{} | {}: {}".format(mixture_key, key, mixtures[mixture_key][key]))

                if len(block_metadata) > 0:
                    block_metadata = "{}  |  {}".format(block_metadata, "{}: {}: {}".format(mixture_key, key, mixtures[mixture_key][key]))
                else:
                    block_metadata = "{}: {}: {}".format(mixture_key, key, mixtures[mixture_key][key])

            print("\n")

        return mixtures, block_metadata

    def convert_lora_dimensions(self, max_dimension, lora):
        new_lora = {}

        for key in lora.keys():
            temp_weights = lora[key]

            if temp_weights.shape[0] < temp_weights.shape[1]:
                if temp_weights.shape[0] < max_dimension:
                    padding = torch.zeros([max_dimension, temp_weights.shape[1]])
                    padding[:temp_weights.shape[0],:] = temp_weights
                    new_lora[key] = padding
                else:
                    new_lora[key] = temp_weights
            else:
                if temp_weights.shape[1] < max_dimension:
                    padding = torch.zeros([temp_weights.shape[0], max_dimension])
                    padding[:,:temp_weights.shape[1]] = temp_weights
                    new_lora[key] = padding
                else:
                    new_lora[key] = temp_weights
        lora = None

        return new_lora

    def merge_multiple_loras(self, save_folder, model_prefix, seed, num_output, lora_1, lora_2, lora_3, lora_4, lora_5, lora_6, lora_7, lora_8, save_metadata=True):
        print("Save_folder: {}".format(save_folder))
        os.makedirs(save_folder, exist_ok = True)

        temp_loras = {}
        metadata = {"loras": "{} and {} and {} and {} and {} and {} and {} and {}".format(lora_1, lora_2, lora_3, lora_4, lora_5, lora_6, lora_7, lora_8)}
        metadata["seed"] = "{}".format(seed)
        metadata["num_output"] = "{}".format(num_output)

        if lora_1 != "None":
            print(lora_1)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_1, 1.0, "all")
            temp_loras["1"] = {"lora_weights": lora_weights}

        if lora_2 != "None":
            print(lora_2)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_2, 1.0, "all")
            temp_loras["2"] = {"lora_weights": lora_weights}

        if lora_3 != "None":
            print(lora_3)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_3, 1.0, "all")
            temp_loras["3"] = {"lora_weights": lora_weights}

        if lora_4 != "None":
            print(lora_4)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_4, 1.0, "all")
            temp_loras["4"] = {"lora_weights": lora_weights}

        if lora_5 != "None":
            print(lora_5)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_5, 1.0, "all")
            temp_loras["5"] = {"lora_weights": lora_weights}

        if lora_6 != "None":
            print(lora_6)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_6, 1.0, "all")
            temp_loras["6"] = {"lora_weights": lora_weights}

        if lora_7 != "None":
            print(lora_7)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_7, 1.0, "all")
            temp_loras["7"] = {"lora_weights": lora_weights}

        if lora_8 != "None":
            print(lora_8)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_8, 1.0, "all")
            temp_loras["8"] = {"lora_weights": lora_weights}

        loras = {}
        max_dimension = 0

        for lora_key in temp_loras.keys():
            # print(lora_key)
            loras[lora_key] = {"lora_weights": {}}

            for key in temp_loras[lora_key]["lora_weights"].keys():
                new_key = key.replace("transformer.", "diffusion_model.")
                loras[lora_key]["lora_weights"][new_key] = temp_loras[lora_key]["lora_weights"][key]

                temp_dimension = min(loras[lora_key]["lora_weights"][new_key].shape[0], loras[lora_key]["lora_weights"][new_key].shape[1])

                if temp_dimension > max_dimension:
                    max_dimension = temp_dimension

        merge_mixtures, block_metadata = self.get_mixtures(seed, num_output, loras.keys())
        metadata["max_dimension"] = "{}".format(max_dimension)

        print("Max Dimension: {}".format(max_dimension))

        save_message = ""

        for mixture_key in merge_mixtures:
            new_lora = {}
            output_filename = os.path.join(save_folder, "{}_{:05}.safetensors".format(model_prefix, int(mixture_key)))

            metadata["merge_mixture"] = "{}".format(merge_mixtures[mixture_key])
            # metadata["block_metadata"] = "{}".format(block_metadata[int(mixture_key)])

            for lora_key in loras.keys():
                mixture_single_blocks = merge_mixtures[mixture_key][lora_key]["single"]
                mixture_double_blocks = merge_mixtures[mixture_key][lora_key]["double"]

                for key in loras[lora_key]["lora_weights"].keys():
                    temp_strings = str(key).split('.')
                    temp_block_num = int(temp_strings[2])

                    if temp_strings[1] == "single_blocks":
                        if temp_block_num in mixture_single_blocks:
                            new_lora[key] = loras[lora_key]["lora_weights"][key]
                        continue

                    if temp_strings[1] == "double_blocks":
                        if temp_block_num in mixture_double_blocks:
                            new_lora[key] = loras[lora_key]["lora_weights"][key]

            new_lora = self.convert_lora_dimensions(max_dimension, new_lora)

            if not save_metadata:
                metadata = None

            print("Saving Model To: {}...".format(output_filename))
            utils.save_torch_file(new_lora, output_filename, metadata=metadata)
            print("Saving Model To: {}...Done.".format(output_filename))

            save_message = "{}\n{}".format(save_message, "Weights Saved To: {}".format(output_filename))

            new_lora = None
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

        return {"ui": {"tags": ["save_message"]}}

class WarpedHunyuanMultiLoraMixerExt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_folder": ("STRING", {"default": get_default_output_folder()}),
                "model_prefix": ("STRING", {"default": "new_model_hy"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "num_output": ("INT", {"default": 1, "min": 1, "max": 100}),
                "lora_1": (['None'] + get_lora_list(),),
                "strength_1": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "lora_2": (['None'] + get_lora_list(),),
                "strength_2": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "lora_3": (['None'] + get_lora_list(),),
                "strength_3": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "lora_4": (['None'] + get_lora_list(),),
                "strength_4": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "lora_5": (['None'] + get_lora_list(),),
                "strength_5": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "lora_6": (['None'] + get_lora_list(),),
                "strength_6": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "lora_7": (['None'] + get_lora_list(),),
                "strength_7": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "lora_8": (['None'] + get_lora_list(),),
                "strength_8": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "merge_multiple_loras"
    CATEGORY = "Warped/HunyuanTools"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

    def convert_key_format(self, key: str) -> str:
        """Standardize LoRA key format by removing prefixes."""
        prefixes = ["diffusion_model.", "transformer."]
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        return key

    def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load and filter a single LoRA model."""
        if not lora_name or strength == 0:
            return {}, {}

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        # Load the LoRA weights
        lora_weights = utils.load_torch_file(lora_path)

        return lora_weights

    def get_random_key(self, keys):
        if len(keys) > 0:
            random_key = random.randint(0, len(keys) - 1)
            # print("Keys: {}  |  Random Key: {}".format(random_key))
            return keys[random_key]

        return -1

    def get_mixtures(self, seed, num_output, lora_keys, block_types):
        random.seed(seed)
        mixtures = {}

        for i in range(num_output):
            mixtures["{}".format(i + 1)] = {}

        single_block_loras = []
        double_block_loras = []

        for key in lora_keys:
            for mixture_key in mixtures.keys():
                mixtures[mixture_key][key] = {"single": [], "double": []}

                if block_types[key]["has_single_blocks"]:
                    single_block_loras.append(int(key))

                if block_types[key]["has_double_blocks"]:
                    double_block_loras.append(int(key))

        for mixture_key in mixtures.keys():
            if len(single_block_loras) > 0:
                for j in range(40):
                    random_key = self.get_random_key(single_block_loras)
                    temp_key = "{}".format(random_key)
                    mixtures[mixture_key][temp_key]["single"].append(j)

            if len(double_block_loras) > 0:
                for j in range(20):
                    random_key = self.get_random_key(double_block_loras)
                    temp_key = "{}".format(random_key)
                    mixtures[mixture_key][temp_key]["double"].append(j)

            i += 1

        print("\nMixtures\n")

        block_metadata = ""

        for mixture_key in mixtures.keys():
            for key in mixtures[mixture_key]:
                print("{} | {}: {}".format(mixture_key, key, mixtures[mixture_key][key]))

                if len(block_metadata) > 0:
                    block_metadata = "{}  |  {}".format(block_metadata, "{}: {}: {}".format(mixture_key, key, mixtures[mixture_key][key]))
                else:
                    block_metadata = "{}: {}: {}".format(mixture_key, key, mixtures[mixture_key][key])

            print("\n")

        return mixtures, block_metadata

    def determine_lora_block_types(self, loras):
        block_types = {}

        for lora_key in loras.keys():
            block_types[lora_key] = { "has_single_blocks": False, "has_double_blocks": False }

            for key in loras[lora_key]["lora_weights"].keys():
                if "single_blocks" in key:
                    block_types[lora_key]["has_single_blocks"] = True
                elif "double_blocks" in key:
                    block_types[lora_key]["has_double_blocks"] = True

                if block_types[lora_key]["has_single_blocks"] and block_types[lora_key]["has_double_blocks"]:
                    break

        return block_types

    def convert_lora_dimensions(self, max_dimension, lora):
        new_lora = {}

        for key in lora.keys():
            temp_weights = lora[key]

            if temp_weights.shape[0] < temp_weights.shape[1]:
                if temp_weights.shape[0] < max_dimension:
                    padding = torch.zeros([max_dimension, temp_weights.shape[1]])
                    padding[:temp_weights.shape[0],:] = temp_weights
                    new_lora[key] = padding
                else:
                    new_lora[key] = temp_weights
            else:
                if temp_weights.shape[1] < max_dimension:
                    padding = torch.zeros([temp_weights.shape[0], max_dimension])
                    padding[:,:temp_weights.shape[1]] = temp_weights
                    new_lora[key] = padding
                else:
                    new_lora[key] = temp_weights
        lora = None

        return new_lora

    def merge_multiple_loras(self, save_folder, model_prefix, seed, num_output, lora_1, strength_1, lora_2, strength_2, lora_3, strength_3, lora_4, strength_4,
                            lora_5, strength_5, lora_6, strength_6, lora_7, strength_7, lora_8, strength_8, save_metadata=True):
        print("Save_folder: {}".format(save_folder))
        os.makedirs(save_folder, exist_ok = True)

        temp_loras = {}
        metadata = {"loras": "{} and {} and {} and {} and {} and {} and {} and {}".format(lora_1, lora_2, lora_3, lora_4, lora_5, lora_6, lora_7, lora_8)}
        metadata["strengths"] = "{} and {} and {} and {} and {} and {} and {} and {}".format(strength_1, strength_2, strength_3, strength_4, strength_5, strength_6, strength_7, strength_8)
        metadata["seed"] = "{}".format(seed)
        metadata["num_output"] = "{}".format(num_output)

        if (lora_1 != "None") and (strength_1 > 0.0):
            print(lora_1)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_1, 1.0, "all")
            temp_loras["1"] = {"lora_weights": lora_weights, "strength": strength_1}

        if (lora_2 != "None") and (strength_2 > 0.0):
            print(lora_2)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_2, 1.0, "all")
            temp_loras["2"] = {"lora_weights": lora_weights, "strength": strength_2}

        if (lora_3 != "None") and (strength_3 > 0.0):
            print(lora_3)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_3, 1.0, "all")
            temp_loras["3"] = {"lora_weights": lora_weights, "strength": strength_3}

        if (lora_4 != "None") and (strength_4 > 0.0):
            print(lora_4)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_4, 1.0, "all")
            temp_loras["4"] = {"lora_weights": lora_weights, "strength": strength_4}

        if (lora_5 != "None") and (strength_5 > 0.0):
            print(lora_5)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_5, 1.0, "all")
            temp_loras["5"] = {"lora_weights": lora_weights, "strength": strength_5}

        if (lora_6 != "None") and (strength_6 > 0.0):
            print(lora_6)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_6, 1.0, "all")
            temp_loras["6"] = {"lora_weights": lora_weights, "strength": strength_6}

        if (lora_7 != "None") and (strength_7 > 0.0):
            print(lora_7)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_7, 1.0, "all")
            temp_loras["7"] = {"lora_weights": lora_weights, "strength": strength_7}

        if (lora_8 != "None") and (strength_8 > 0.0):
            print(lora_8)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_8, 1.0, "all")
            temp_loras["8"] = {"lora_weights": lora_weights, "strength": strength_8}

        loras = {}
        max_dimension = 0

        for lora_key in temp_loras.keys():
            # print(lora_key)
            loras[lora_key] = {"lora_weights": {}, "strength": temp_loras[lora_key]["strength"]}

            for key in temp_loras[lora_key]["lora_weights"].keys():
                new_key = key.replace("transformer.", "diffusion_model.")
                loras[lora_key]["lora_weights"][new_key] = temp_loras[lora_key]["lora_weights"][key]

                temp_dimension = min(loras[lora_key]["lora_weights"][new_key].shape[0], loras[lora_key]["lora_weights"][new_key].shape[1])

                if temp_dimension > max_dimension:
                    max_dimension = temp_dimension

        block_types = self.determine_lora_block_types(loras)
        merge_mixtures, block_metadata = self.get_mixtures(seed, num_output, loras.keys(), block_types)

        metadata["max_dimension"] = "{}".format(max_dimension)
        metadata["block_types"] = "{}".format(block_types)

        print("Max Dimension: {}".format(max_dimension))

        save_message = ""

        for mixture_key in merge_mixtures:
            new_lora = {}
            output_filename = os.path.join(save_folder, "{}_{:05}.safetensors".format(model_prefix, int(mixture_key)))

            metadata["merge_mixture"] = "{}".format(merge_mixtures[mixture_key])
            # metadata["block_metadata"] = "{}".format(block_metadata[int(mixture_key)])

            for lora_key in loras.keys():
                mixture_single_blocks = merge_mixtures[mixture_key][lora_key]["single"]
                mixture_double_blocks = merge_mixtures[mixture_key][lora_key]["double"]

                for key in loras[lora_key]["lora_weights"].keys():
                    temp_strings = str(key).split('.')
                    temp_block_num = int(temp_strings[2])

                    if temp_strings[1] == "single_blocks":
                        if temp_block_num in mixture_single_blocks:
                            new_lora[key] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"])
                        continue

                    if temp_strings[1] == "double_blocks":
                        if temp_block_num in mixture_double_blocks:
                            new_lora[key] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"])

            new_lora = self.convert_lora_dimensions(max_dimension, new_lora)

            if not save_metadata:
                metadata = None

            print("Saving Model To: {}...".format(output_filename))
            utils.save_torch_file(new_lora, output_filename, metadata=metadata)
            print("Saving Model To: {}...Done.".format(output_filename))

            save_message = "{}\n{}".format(save_message, "Weights Saved To: {}".format(output_filename))

            new_lora = None
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

        return {"ui": {"tags": ["save_message"]}}

class WarpedWanLoraMerge:
    def __init__(self):
        self.base_output_dir = get_default_wan_output_folder()
        os.makedirs(self.base_output_dir, exist_ok = True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_path": ("STRING", {"default": get_default_wan_output_path()}),
                "lora_1": (['None'] + get_lora_list(),),
                "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "lora_2": (['None'] + get_lora_list(),),
                "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "merge_multiple_loras"
    CATEGORY = "Warped/WanTools"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

    def convert_key_format(self, key: str) -> str:
        """Standardize LoRA key format by removing prefixes."""
        prefixes = ["diffusion_model.", "transformer."]
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        return key

    def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load and filter a single LoRA model."""
        if not lora_name or strength == 0:
            return {}, {}

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        # Load the LoRA weights
        lora_weights = utils.load_torch_file(lora_path)

        return lora_weights

    def merge_multiple_loras(self, save_path, lora_1, strength_1, lora_2, strength_2, save_metadata=True):
        """Load and apply multiple LoRA models."""
        temp_loras = {}
        metadata = {"loras": "{} and {}".format(lora_1, lora_2)}
        metadata["strengths"] = "{} and {}".format(strength_1, strength_2)

        if lora_1 != "None" and strength_1 != 0:
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_1, 1.0, "all")
            temp_loras["1"] = {"lora_weights": lora_weights, "strength": strength_1}

        if lora_2 != "None" and strength_2 != 0:
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_2, 1.0, "all")
            temp_loras["2"] = {"lora_weights": lora_weights, "strength": strength_2}

        loras = {}

        for lora_key in temp_loras.keys():
            loras[lora_key] = {"lora_weights": {}, "strength": temp_loras[lora_key]["strength"]}

            for key in temp_loras[lora_key]["lora_weights"].keys():
                # new_key = key.replace("transformer.", "diffusion_model.")
                loras[lora_key]["lora_weights"][key] = temp_loras[lora_key]["lora_weights"][key]

        new_lora = {}

        for lora_key in loras.keys():
            for key in loras[lora_key]["lora_weights"].keys():
                if not key in new_lora.keys():
                    new_lora[key] = None
                print("Lora: {}  | Key: {}  |  Shape: {}".format(lora_key, key, loras[lora_key]["lora_weights"][key].shape))

        # print_it = True
        #
        # i = 0

        # Merge The Weighted Key Weights
        for key in new_lora.keys():
            for lora_key in loras.keys():
                if key in loras[lora_key]["lora_weights"].keys():
                    if not new_lora[key] is None:
                        temp_weights = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"])
                        #
                        # if print_it:
                        #     print("\n--------------------------------------------------------------------")
                        #     print("Before: New Lora Shape: {}  |  Temp Weights Shape: {}".format(new_lora[key].shape, temp_weights.shape))
                        #     print("--------------------------------------------------------------------\n")

                        if new_lora[key].shape[0] < new_lora[key].shape[1]:
                            if temp_weights.shape[0] < new_lora[key].shape[0]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([new_lora[key].shape[0], new_lora[key].shape[1]])
                                padding[:temp_weights.shape[0],:] = temp_weights
                                temp_weights = padding
                            elif temp_weights.shape[0] > new_lora[key].shape[0]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([temp_weights.shape[0], temp_weights.shape[1]])
                                padding[:new_lora[key].shape[0],:] = new_lora[key]
                                new_lora[key] = padding
                        else:
                            if temp_weights.shape[1] < new_lora[key].shape[1]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([new_lora[key].shape[0], new_lora[key].shape[1]])
                                padding[:,:temp_weights.shape[1]] = temp_weights
                                temp_weights = padding
                            elif temp_weights.shape[1] > new_lora[key].shape[1]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([temp_weights.shape[0], temp_weights.shape[1]])
                                padding[:,:new_lora[key].shape[1]] = new_lora[key]
                                new_lora[key] = padding

                        try:
                            # new_lora[key] = ((new_lora[key] + temp_weights) / 2.0)
                            new_lora[key] = torch.add(new_lora[key], temp_weights)

                            # if print_it:
                            #     if (i + 1) >= 10:
                            #         print_it = False
                            #
                            #     i += 1
                            #     print("\n--------------------------------------------------------------------")
                            #     print("After: New Lora Shape: {}".format(new_lora[key].shape))
                            #     print("--------------------------------------------------------------------\n")
                                # print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                                # print("After: New Lora Key Tensor: {}".format(new_lora[key]))
                                # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                        except Exception as e:
                            raise(e)
                    else:
                        new_lora[key] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"])

        if not save_metadata:
            metadata = None

        utils.save_torch_file(new_lora, save_path, metadata=metadata)

        save_message = "Weights Saved To: {}".format(save_path)

        return {"ui": {"tags": [save_message]}}

class WarpedCreateSpecialImageBatch:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    { "image": ("IMAGE", ),
                      "color": ("STRING", {"default": "#000000"}),
                      "batch_size": ("INT", {"default": 1, "min": 1, "max": 1001, "step": 4}),
                      "all_same_image": ("BOOLEAN", {"default": False}),
                    }
                }
    RETURN_TYPES = ("IMAGE", "INT", )
    RETURN_NAMES = ("image", "num_images", )
    FUNCTION = "generate"

    CATEGORY = "Warped/Image"

    def generate(self, image, color="#000000", batch_size=1, all_same_image=False):
        image_color = ImageColor.getrgb(color)

        temp_image = tensor2pilSwap(image)
        temp_image = pil2tensorSwap(temp_image[0])
        temp_image = tensor2pilSwap(temp_image)
        temp_image = temp_image[0]

        if not all_same_image:
            dummy_image = temp_image.copy()
            dummy_image.paste((0,0,0), (0, 0, dummy_image.size[0], dummy_image.size[1]))

            image_batch = [temp_image]

            i = 0
            while i < (batch_size - 1):
                image_batch.append(dummy_image.copy())

                i += 1

            intermediate_images = pil2tensorSwap(image_batch)

            final_images = None

            print_it = True
            for image in intermediate_images:
                image = image.unsqueeze(0)
                if not final_images is None:
                    final_images = torch.cat((final_images, image), dim=0)

                    # if print_it:
                    #     print_it = False
                    #     print(final_images)
                else:
                    final_images = image

            return (final_images, batch_size, )

        image_batch = [temp_image]

        i = 0
        while i < (batch_size - 1):
            image_batch.append(temp_image.copy())

            i += 1

        intermediate_images = pil2tensorSwap(image_batch)

        final_images = None

        print_it = True
        for image in intermediate_images:
            image = image.unsqueeze(0)
            if not final_images is None:
                final_images = torch.cat((final_images, image), dim=0)
            else:
                final_images = image

        return (final_images, batch_size, )

class WarpedCreateSpecialImageBatchExt:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    { "image": ("IMAGE", ),
                      "color": ("STRING", {"default": "#000000"}),
                      "batch_size": ("INT", {"default": 1, "min": 1, "max": 1001, "step": 4}),
                      "all_same_image": ("BOOLEAN", {"default": False}),
                    },
                "optional":
                    {
                      "start_image": ("IMAGE", ),
                    }
                }
    RETURN_TYPES = ("IMAGE", "INT", )
    RETURN_NAMES = ("image", "num_images", )
    FUNCTION = "generate"

    CATEGORY = "Warped/Image"

    def generate(self, image, color="#000000", batch_size=1, all_same_image=False, start_image=None):
        image_color = ImageColor.getrgb(color)

        if len(image.shape) < 4:
            image = image.unsqueeze(0)

        height = image.shape[1]
        width  = image.shape[2]

        temp_image = image.clone().detach()

        if not all_same_image:
            dummy_image = torch.zeros([batch_size - 1, self.height, self.width, 3], dtype=torch.float32, device=self.offload_device)
            final_images = torch.cat((image, dummy_image), 0)

            return (final_images, batch_size, )

        if start_image is None:
            final_images = temp_image
        else:
            final_images = start_image

        i = 0
        while i < (batch_size - 1):
            final_images = torch.cat((final_images, temp_image.clone().detach()), 0)
            i += 1

        return (final_images, batch_size, )

class WarpedCreateSpecialImageBatchExp:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    { "image": ("IMAGE", ),
                      "batch_size": ("INT", {"default": 1, "min": 1, "max": 1001, "step": 4}),
                      "all_same_image": ("BOOLEAN", {"default": False}),
                    }
                }
    RETURN_TYPES = ("IMAGE", "INT", )
    RETURN_NAMES = ("image", "num_images", )
    FUNCTION = "generate"

    CATEGORY = "Warped/Image"

    def generate(self, image, batch_size=1, all_same_image=False):
        if len(image.shape) < 4:
            image = image.unsqueeze(0)

        height = image.shape[1]
        width = image.shape[2]

        if not all_same_image:
            temp_image = torch.ones((batch_size - 1, height, width, image.shape[-1]), device=image.device, dtype=image.dtype) * 0.5
            final_images = torch.cat((image, temp_image), 0)

            return (final_images, batch_size, )

        final_images = image.clone().detach()

        i = 0
        while i < (batch_size - 1):
            final_images = torch.cat((final_images, image.clone().detach()), 0)
            i += 1

        return (final_images, batch_size, )

class WarpedCreateEmptyImageBatch:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    { "batch_size": ("INT", {"default": 1, "min": 1, "max": 1001, "step": 4}),
                      "width": ("INT", {"default": 320, "min": 256, "max": 4096, "step": 16}),
                      "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 16}),
                    }
                }
    RETURN_TYPES = ("IMAGE", "INT", )
    RETURN_NAMES = ("image", "num_images", )
    FUNCTION = "generate"

    CATEGORY = "Warped/Image"

    def generate(self, batch_size, width, height):
        color="#000000"
        image_color = ImageColor.getrgb(color)

        temp_image = Image.new(mode="RGB", size=(width, height))
        temp_image = pil2tensorSwap(temp_image)

        if len(temp_image.shape) < 4:
            temp_image = temp_image.unsqueeze(0)

        final_images = None
        count = 0

        while count < batch_size:
            if not final_images is None:
                final_images = torch.cat((final_images, temp_image.clone().detach()), dim=0)
            else:
                final_images = temp_image.clone().detach()

            count += 1

        return (final_images, batch_size, )

class WarpedCreateEmptyLatentBatch:
    def __init__(self, device="cpu"):
        self.device = device
        self.offload_device = get_offload_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    { "batch_size": ("INT", {"default": 1, "min": 1, "max": 1001, "step": 4}),
                      "width": ("INT", {"default": 320, "min": 256, "max": 4096, "step": 16}),
                      "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 16}),
                    }
                }
    RETURN_TYPES = ("LATENT", "INT", )
    RETURN_NAMES = ("latents", "num_images", )
    FUNCTION = "generate"

    CATEGORY = "Warped/Latent"

    def generate(self, batch_size, width, height):
        temp_latent = torch.zeros([1, 16, int(((batch_size - 1) / 4) + 1), int(height // 8), int(width // 8)], dtype=torch.float32, device=self.offload_device)

        print("Empty Latent Batch Shape: {}".format(temp_latent.shape))

        if len(temp_latent.shape) < 5:
            temp_latent = temp_latent.unsqueeze(0)

        return ({"samples": temp_latent}, batch_size, )

class WarpedSamplerCustomAdv:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE", ),
                    "vae": ("VAE", ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "guider": ("GUIDER", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "enc_tile_size": ("INT", {"default": 128, "min": 64, "max": 4096, "step": 64}),
                    "enc_overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                    "enc_temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to encode at a time."}),
                    "enc_temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap."}),
                    "dec_tile_size": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 32}),
                    "dec_overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                    "dec_temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to decode at a time."}),
                    "dec_temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap."}),
                    "skip_frames": ("INT", {"default": 0, "min": 0, "max": 32, "step": 4}),
                    "noise_scale": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                    },
                "optional":
                    {"scaling_strength": ("FLOAT", {"default": 1.0}),
                    "output_latents": ("BOOLEAN", {"default": False}),
                    }
                }

    RETURN_TYPES = ("IMAGE", "LATENT", "INT", "STRING", )
    RETURN_NAMES = ("images", "latents", "seed", "generation_status", )

    FUNCTION = "sample"

    CATEGORY = "Warped/Sampling"

    def sample(self, image, vae, seed, guider, sampler, sigmas, enc_tile_size, enc_overlap, enc_temporal_size, enc_temporal_overlap,
                    dec_tile_size, dec_overlap, dec_temporal_size, dec_temporal_overlap, skip_frames, noise_scale, scaling_strength=1.0, output_latents=False):
        self.device = mm.get_torch_device()
        self.offload_device = get_offload_device()
        mm.unload_all_models()
        gc.collect()
        time.sleep(1)

        self.vae = vae
        self.seed = seed
        self.guider = guider
        self.sampler = sampler
        self.sigmas = sigmas
        self.noise_scale = noise_scale
        self.enc_tile_size = enc_tile_size
        self.enc_overlap = enc_overlap
        self.enc_temporal_size = enc_temporal_size
        self.enc_temporal_overlap = enc_temporal_overlap
        self.dec_tile_size = dec_tile_size
        self.dec_overlap = dec_overlap
        self.dec_temporal_size = dec_temporal_size
        self.dec_temporal_overlap = dec_temporal_overlap
        self.g_output = {}

        callback = self.setup_callbacks()
        disable_pbar = not utils.PROGRESS_BAR_ENABLED

        if len(image.shape) < 4:
            image = image.unsqueeze(0)

        num_frames = image.shape[0]

        self.width = image.shape[2]
        self.height = image.shape[1]
        print("\nWidth is {}  |  Height is {}".format(self.width, self.height))

        generation_status = ""

        latents, noise_latents = self.initialize_frames(image)

        print("-------------------------------------------------------------------------------------------")
        print("WarpedSamplerCustomAdv: Latents Shape: {}  |  Noise Latents Shape: {}".format(latents.shape, noise_latents.shape))
        print("-------------------------------------------------------------------------------------------")

        output_images = None
        output_images_latents = None
        interrupted = False

        try:
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(0.1)

            latent = {"samples": latents}

            noise_mask = None
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"]

            print("Noise Shape: {}  |  Latents Shape: {}".format(noise_latents.shape, latents.shape))
            print("WarpedSamplerCustomAdv: Generating {} Frames in {} Latents....".format(num_frames, latents.shape[2]))

            samples = guider.sample(noise_latents, latents, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=self.seed)
            samples = samples.to(mm.intermediate_device())

            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

            if len(samples.shape) < 5:
                samples = samples.unsqueeze(0)

            samples = samples.clone().detach() / scaling_strength

            print("WarpedSamplerCustomAdv: Decoding Latents...")
            output_images = self.decode_tiled(samples)
            samples = None

            output_images = self.process_skip_images(output_images, skip_frames)
            print("WarpedSamplerCustomAdv: Decoded Images Shape: {}".format(output_images.shape[0]))

            if output_latents:
                output_images_latents = self.encode_tiled(output_images)

            print("WarpedSamplerCustomAdv: Generating {} frames in {} latents...Done.".format(num_frames, latents.shape[2]))

            samples = None
            latents = None

            mm.soft_empty_cache()
            gc.collect()
            time.sleep(0.1)

            print("*******************************************")
            print("****** WarpedSamplerCustomAdv: Total Images Generated {}".format(output_images.shape[0]))
            print("*******************************************\n")

            if len(output_images.shape) < 4:
                output_images = output_images.unsqueeze(0)

            generation_status = "****** WarpedSamplerCustomAdv: Total Images Generated {} ******".format(output_images.shape[0])

            interrupted = False
        except mm.InterruptProcessingException as ie:
            interrupted = True
            print(f"\nWarpedSamplerCustomAdv: Processing Interrupted.")
            print("WarpedSamplerCustomAdv: Returning only partial results (if any).\n If zero images generated, a blank yellow image will be returned.\n")
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

            generation_status = f"\nWarpedSamplerCustomAdv: Processing Interrupted."

            traceback.print_tb(ie.__traceback__, limit=99, file=sys.stdout)

            raise mm.InterruptProcessingException(f"WarpedSamplerCustomAdv: Processing Interrupted.")

        except BaseException as e:
            print(f"\nWarpedSamplerCustomAdv: Exception During Processing: {str(e)}")
            print("WarpedSamplerCustomAdv: Returning only partial results (if any).\n If zero images generated, a blank red image will be returned.\n")
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

            generation_status = f"WarpedSamplerCustomAdv: Exception During Processing: {str(e)}"
            generation_status = "{}{}".format(generation_status, "WarpedSamplerCustomAdv: Returning only partial results (if any).\nIf zero images generated, a blank red image will be returned.")

            traceback.print_tb(e.__traceback__, limit=99, file=sys.stdout)

            pass

        callback = None
        guider.model_patcher.model.to(get_offload_device())

        latent = None
        latent_image = None
        noise_mask = None
        samples = None

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        time.sleep(1)

        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        image = None

        if interrupted:
            temp_image = Image.new('RGB', (self.width, self.height), color = 'yellow')
            image = pil2tensorSwap(temp_image)
            output_images = image
        elif output_images is None:
            temp_image = Image.new('RGB', (self.width, self.height), color = 'red')
            image = pil2tensorSwap(temp_image)
            output_images = image

        if output_images_latents is None:
            output_images_latents = torch.zeros([1, 16, 1, self.height // 8, self.width // 8], dtype=torch.float32, device=self.offload_device)

        return (output_images, {"samples": output_images_latents}, self.seed, generation_status,)

    def generate_noise(self, input_latent, generator=None):
        latent_image = input_latent["samples"]
        return warped_prepare_noise(latent_image, self.seed, generator=generator)

    def process_skip_images(self, frames, skip_count):
        if len(frames.shape) < 4:
            frames = frames.unsqueeze(0)

        num_frames = frames.shape[0]

        image_batches_tuple = torch.split(frames, 1, dim=0)
        image_batches_split = [item for item in image_batches_tuple]

        new_video = None
        i = 0

        while i < len(image_batches_split):
            if i < skip_count:
                i += 1
                continue

            if not new_video is None:
                new_video = torch.cat((new_video, image_batches_split[i]), 0)
            else:
                new_video = image_batches_split[i]

            i += 1

        return new_video

    def get_blank_image(self, length=1):
        new_image = torch.zeros([length, self.height, self.width, 3], dtype=torch.float32, device=self.offload_device)
        return new_image

    def get_new_noise(self, length):
        new_noise = torch.zeros([length, self.height, self.width, 3], dtype=torch.float32, device=self.offload_device)

        new_noise = self.encode_tiled(new_noise)

        new_noise = comfy.sample.fix_empty_latent_channels(self.guider.model_patcher, new_noise)

        if len(new_noise) < 5:
            new_latent = new_noise.unsqueeze(0)

        new_noise = self.generate_noise({"samples": new_noise})

        return new_noise

    def pad_noise(self, latent, num_frames=1):
        pad_frames = torch.zeros([1, 16, num_frames, self.height, self.width], dtype=torch.float32, device=self.offload_device)
        pad_frames = torch.cat((latent, pad_frames), 2)

        return pad_frames

    def setup_callbacks(self):
        callback = latent_preview.prepare_callback(self.guider.model_patcher, self.sigmas.shape[-1] - 1, self.g_output)

        return callback

    def encode(self, images):
        if len(images.shape) < 4:
            images = images.unsqueze(0)

        encoded_data = partial_encode_basic(self.vae, images)

        if len(encoded_data.shape) < 5:
            encoded_data.unsqueeze(0)

        return encoded_data

    def encode_as_batched(self, images):
        if len(images.shape) < 4:
            images = images.unsqueze(0)

        print("WarpedSamplerCustomAdv: latents Shape Before Split: {}".format(latents.shape))
        image_batches_tuple = torch.split(images, self.enc_split_sizes, dim=0)
        image_batches_split = [item for item in image_batches_tuple]

        if not (image_batches_split is None) and (len(image_batches_split) > 0):
            print("WarpedSamplerCustomAdv: Splits Length: {}".format(len(image_batches_split)))
            for entry in image_batches_split:
                print(entry.shape)
            print("WarpedSamplerCustomAdv: Splits End")
        else:
            print("++++++ WarpedSamplerCustomAdv: Empty Results Splits ++++++")
            return None

        result_images = None

        print("WarpedSamplerCustomAdv: Encoding Batch Latents...")
        for entry in image_batches_split:
            encoded_data = self.encode(entry)

            if len(encoded_data.shape) < 4:
                encoded_data = encoded_data.unsqueeze(0)

            print("WarpedSamplerCustomAdv: Entry Shape: {}  |  Encoded Entry Shape: {}".format(entry.shape, encoded_data.shape))

            if not result_images is None:
                result_images = torch.cat((result_images, encoded_data), 0)
            else:
                result_images = decoded_data

        print("WarpedSamplerCustomAdv: Encoded Images Shape: {}".format(result_images.shape))
        print("WarpedSamplerCustomAdv: Encoding Batch Latents...Done.")

        return result_images

    def encode_tiled(self, images):
        if len(images.shape) < 4:
            images = images.unsqueze(0)

        encoded_data = partial_encode_tiled(self.vae, images, self.enc_tile_size, self.enc_overlap, self.enc_temporal_size, self.enc_temporal_overlap)

        if len(encoded_data.shape) < 5:
            encoded_data.unsqueeze(0)

        return encoded_data

    def decode_tiled(self, latents):
        decoded_data = partial_decode_tiled(self.vae, latents, self.dec_tile_size, self.dec_overlap, self.dec_temporal_size, self.dec_temporal_overlap)

        if len(decoded_data.shape) < 4:
            decoded_data.unsqueeze(0)

        return decoded_data

    def initialize_noise(self, frame_count, clear_cache=True):
        noise_latents_full = torch.zeros([1, 16, int(frame_count), self.height // 8, self.width // 8], dtype=torch.float32, device=self.offload_device)
        print("WarpedSamplerCustomAdv: Encoded noise_latents_full Shape: {}".format(noise_latents_full.shape))

        if Decimal(self.noise_scale).compare(Decimal(0.00)) != 0:
            noise_latents_full = warped_prepare_noise(noise_latents_full, self.seed)
            print("WarpedSamplerCustomAdv: noise_latents_full Shape: {}".format(noise_latents_full.shape))

            noise_latents_full = torch.mul(noise_latents_full, self.noise_scale)

        if len(noise_latents_full.shape) < 5:
            noise_latents_full.unsqueeze(0)

        noise_latents_full = noise_latents_full.to(dtype=torch.float32, device=self.offload_device)

        if clear_cache:
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(0.1)

        return noise_latents_full

    def initialize_frames(self, image):
        if len(image.shape) < 4:
            image = image.unsqueeze(0)

        print("WarpedSamplerCustomAdv: Decoded latents_full Shape: {}".format(image.shape))
        latents_full = self.encode_tiled(image)

        if len(latents_full.shape) < 5:
            latents_full.unsqueeze(0)

        print("WarpedSamplerCustomAdv: Encoded latents_full Shape: {}".format(latents_full.shape))

        latents_full = latents_full.to(dtype=torch.float32, device=self.offload_device)

        noise_latents_full = self.initialize_noise(latents_full.shape[2])

        mm.soft_empty_cache()
        gc.collect()
        time.sleep(0.1)

        return latents_full, noise_latents_full

class WarpedSamplerCustomAdvLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"latent": ("LATENT", ),
                    "vae": ("VAE", ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "guider": ("GUIDER", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "dec_tile_size": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 32}),
                    "dec_overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                    "dec_temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to decode at a time."}),
                    "dec_temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap."}),
                    "skip_frames": ("INT", {"default": 0, "min": 0, "max": 32, "step": 4}),
                    "noise_scale": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                    },
                "optional":
                    {"scaling_strength": ("FLOAT", {"default": 1.0}),
                    # "block_swap_args": ("BLOCKSWAPARGS", ),
                    "output_latents": ("BOOLEAN", {"default": False}),
                    }
                }

    RETURN_TYPES = ("IMAGE", "LATENT", "INT", "STRING", )
    RETURN_NAMES = ("images", "latents", "seed", "generation_status", )

    FUNCTION = "sample"

    CATEGORY = "Warped/Sampling"

    def sample(self, latent, vae, seed, guider, sampler, sigmas, dec_tile_size, dec_overlap, dec_temporal_size, dec_temporal_overlap,
                    skip_frames, noise_scale, scaling_strength=1.0, output_latents=False):
        self.device = mm.get_torch_device()
        self.offload_device = get_offload_device()
        mm.unload_all_models()
        gc.collect()
        time.sleep(1)

        self.vae = vae
        self.seed = seed
        self.guider = guider
        self.sampler = sampler
        self.sigmas = sigmas
        self.noise_scale = noise_scale
        self.dec_tile_size = dec_tile_size
        self.dec_overlap = dec_overlap
        self.dec_temporal_size = dec_temporal_size
        self.dec_temporal_overlap = dec_temporal_overlap
        self.g_output = {}

        callback = self.setup_callbacks()
        disable_pbar = not utils.PROGRESS_BAR_ENABLED

        latents = latent["samples"]

        if len(latents.shape) < 5:
            latents = latents.unsqueeze (0)

        num_frames = int(((latents.shape[2] - 1) * 4) + 1)

        self.width = latents.shape[4]
        self.height = latents.shape[3]
        print("\nDecoded Width is {}  |  Decoded Height is {}".format(int(self.width * 8), int(self.height * 8)))

        generation_status = ""

        noise_latents = self.initialize_frames(latents)

        print("-------------------------------------------------------------------------------------------")
        print("WarpedSamplerCustomAdvLatent: Latents Shape: {}  |  Noise Latents Shape: {}".format(latents.shape, noise_latents.shape))
        print("-------------------------------------------------------------------------------------------")

        output_images = None
        output_images_latents = None
        interrupted = False

        try:
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(0.1)

            noise_mask = None
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"]

            print("Noise Shape: {}  |  Latents Shape: {}".format(noise_latents.shape, latents.shape))
            print("WarpedSamplerCustomAdvLatent: Generating {} Frames in {} Latents....".format(num_frames, latents.shape[2]))

            samples = guider.sample(noise_latents, latents, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=self.seed)

            samples = samples.to(mm.intermediate_device())

            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

            if len(samples.shape) < 5:
                samples = samples.unsqueeze(0)

            samples = samples.clone().detach() / scaling_strength

            print("WarpedSamplerCustomAdvLatent: Decoding Latents...")
            output_images = self.decode_tiled(samples)
            samples = None

            output_images = self.process_skip_images(output_images, skip_frames)
            print("WarpedSamplerCustomAdvLatent: Decoded Images Shape: {}".format(output_images.shape[0]))

            if output_latents:
                output_images_latents = self.encode_tiled(output_images)

            print("WarpedSamplerCustomAdvLatent: Generating {} frames in {} latents...Done.".format(num_frames, latents.shape[2]))

            samples = None
            latents = None

            mm.soft_empty_cache()
            gc.collect()
            time.sleep(0.1)

            print("*******************************************")
            print("****** WarpedSamplerCustomAdvLatent: Total Images Generated {}".format(output_images.shape[0]))
            print("*******************************************\n")

            if len(output_images.shape) < 4:
                output_images = output_images.unsqueeze(0)

            generation_status = "****** WarpedSamplerCustomAdvLatent: Total Images Generated {} ******".format(output_images.shape[0])

            interrupted = False
        except mm.InterruptProcessingException as ie:
            interrupted = True
            print(f"\nWarpedSamplerCustomAdvLatent: Processing Interrupted.")
            print("WarpedSamplerCustomAdvLatent: Returning only partial results (if any).\n If zero images generated, a blank yellow image will be returned.\n")
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

            generation_status = f"\nWarpedSamplerCustomAdvLatent: Processing Interrupted."

            traceback.print_tb(ie.__traceback__, limit=99, file=sys.stdout)

            raise mm.InterruptProcessingException(f"WarpedSamplerCustomAdvLatent: Processing Interrupted.")

        except BaseException as e:
            print(f"\nWarpedSamplerCustomAdvLatent: Exception During Processing: {str(e)}")
            print("WarpedSamplerCustomAdvLatent: Returning only partial results (if any).\n If zero images generated, a blank red image will be returned.\n")
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

            generation_status = f"WarpedSamplerCustomAdvLatent: Exception During Processing: {str(e)}"
            generation_status = "{}{}".format(generation_status, "WarpedSamplerCustomAdvLatent: Returning only partial results (if any).\nIf zero images generated, a blank red image will be returned.")

            traceback.print_tb(e.__traceback__, limit=99, file=sys.stdout)

            pass

        callback = None
        guider.model_patcher.model.to(get_offload_device())

        latent = None
        latent_image = None
        noise_mask = None
        samples = None

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        time.sleep(1)

        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        image = None

        if interrupted and (output_images is None):
            temp_image = Image.new('RGB', (self.width * 8, self.height * 8), color = 'yellow')
            image = pil2tensorSwap(temp_image)
            output_images = image
        elif output_images is None:
            temp_image = Image.new('RGB', (self.width * 8, self.height * 8), color = 'red')
            image = pil2tensorSwap(temp_image)
            output_images = image

        if output_images_latents is None:
            output_images_latents = torch.zeros([1, 16, 1, self.height, self.width], dtype=torch.float32, device=self.offload_device)

        return (output_images, {"samples": output_images_latents}, self.seed, generation_status,)

    def generate_noise(self, input_latent, generator=None):
        latent_image = input_latent["samples"]
        return warped_prepare_noise(latent_image, self.seed, generator=generator)

    def process_skip_images(self, frames, skip_count):
        if len(frames.shape) < 4:
            frames = frames.unsqueeze(0)

        num_frames = frames.shape[0]

        image_batches_tuple = torch.split(frames, 1, dim=0)
        image_batches_split = [item for item in image_batches_tuple]

        new_video = None
        i = 0

        while i < len(image_batches_split):
            if i < skip_count:
                i += 1
                continue

            if not new_video is None:
                new_video = torch.cat((new_video, image_batches_split[i]), 0)
            else:
                new_video = image_batches_split[i]

            i += 1

        return new_video

    def get_blank_image(self, length=1):
        new_image = torch.zeros([length, self.height, self.width, 3], dtype=torch.float32, device=self.offload_device)
        return new_image

    def get_new_noise(self, length):
        new_noise = torch.zeros([length, self.height, self.width, 3], dtype=torch.float32, device=self.offload_device)

        new_noise = self.encode_tiled(new_noise)

        new_noise = comfy.sample.fix_empty_latent_channels(self.guider.model_patcher, new_noise)

        if len(new_noise) < 5:
            new_latent = new_noise.unsqueeze(0)

        new_noise = self.generate_noise({"samples": new_noise})

        return new_noise

    def pad_noise(self, latent, num_frames=1):
        pad_frames = torch.zeros([1, 16, num_frames, self.height, self.width], dtype=torch.float32, device=self.offload_device)
        pad_frames = torch.cat((latent, pad_frames), 2)

        return pad_frames

    def setup_callbacks(self):
        callback = latent_preview.prepare_callback(self.guider.model_patcher, self.sigmas.shape[-1] - 1, self.g_output)

        return callback

    def decode_tiled(self, latents):
        decoded_data = partial_decode_tiled(self.vae, latents, self.dec_tile_size, self.dec_overlap, self.dec_temporal_size, self.dec_temporal_overlap)

        if len(decoded_data.shape) < 4:
            decoded_data.unsqueeze(0)

        return decoded_data

    def initialize_noise(self, frame_count, clear_cache=True):
        noise_latents_full = torch.zeros([1, 16, int(frame_count), self.height, self.width], dtype=torch.float32, device=self.offload_device)
        print("WarpedSamplerCustomAdvLatent: Encoded noise_latents_full Shape: {}".format(noise_latents_full.shape))

        if Decimal(self.noise_scale).compare(Decimal(0.00)) != 0:
            noise_latents_full = warped_prepare_noise(noise_latents_full, self.seed)
            print("WarpedSamplerCustomAdvLatent: noise_latents_full Shape: {}".format(noise_latents_full.shape))

            noise_latents_full = torch.mul(noise_latents_full, self.noise_scale)

        if len(noise_latents_full.shape) < 5:
            noise_latents_full.unsqueeze(0)

        noise_latents_full = noise_latents_full.to(dtype=torch.float32, device=self.offload_device)

        if clear_cache:
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(0.1)

        return noise_latents_full

    def initialize_frames(self, latents):
        if len(latents.shape) < 5:
            latents = latents.unsqueeze(0)

        print("WarpedSamplerCustomAdvLatent: Encoded latents_full Shape: {}".format(latents.shape))
        latents_full = latents.clone().detach()
        latents_full = latents_full.to(dtype=torch.float32, device=self.offload_device)

        noise_latents_full = self.initialize_noise(latents_full.shape[2])

        mm.soft_empty_cache()
        gc.collect()
        time.sleep(0.1)

        return noise_latents_full

def sd3_patch(model, shift, multiplier=1000):
    m = model.clone()

    sampling_base = model_sampling.ModelSamplingDiscreteFlow
    sampling_type = model_sampling.CONST

    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass

    result_model_sampling = ModelSamplingAdvanced(model.model.model_config)
    result_model_sampling.set_parameters(shift=shift, multiplier=multiplier)
    m.add_object_patch("model_sampling", result_model_sampling)
    return m

class WarpedCreateSpecialImageBatchFromVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"video_path": ("STRING", {"default": ""}),
                    "last_n_frames": ("INT", {"default": 16}),
                    "num_frames": ("INT", {"default": 61, "min": 5, "max": 1000001, "step": 4}),
                    },
                }
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", )
    RETURN_NAMES = ("image_batch", "first_image", "num_frames",)
    FUNCTION = "generate"

    CATEGORY = "Warped/Video"

    def generate(self, video_path, last_n_frames, num_frames):
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('WarpedCreateSpecialImageBatchFromVideo: width = %d' % width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('WarpedCreateSpecialImageBatchFromVideo: height = %d' % height)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('WarpedCreateSpecialImageBatchFromVideo: length = %d' % length)

        batched_images = None
        first_image = None
        last_image = None
        frame_count = 0
        starting_index = int(length - last_n_frames - 1)

        if starting_index < 0:
            starting_index = 0

        if ((starting_index + num_frames) > length) or (num_frames == 0):
            temp_num_frames = length - starting_index

        try:
            skip = 0

            if starting_index > 0:
                skip = length - starting_index

            if starting_index > 0:
                while(cap.isOpened()) and (skip < starting_index):
                    _, frameorig = cap.read()
                    skip += 1

            while (cap.isOpened()) and (frame_count < temp_num_frames):
                frame_count += 1

                # Take each frame
                _, frameorig = cap.read()

                color_coverted = cv2.cvtColor(frameorig, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(color_coverted)

                if first_image is None:
                    first_image_pil = pil_image.copy()
                    first_image = pil2tensorSwap(first_image_pil)

                last_image_pil = pil_image.copy()
                temp_image = pil2tensorSwap(pil_image)
                last_image = pil2tensorSwap(last_image_pil)

                if len(temp_image.shape) < 4:
                    temp_image = temp_image.unsqueeze(0)

                if not batched_images is None:
                    batched_images = torch.cat((batched_images, temp_image), 0)
                else:
                    batched_images = temp_image
        except:
            print("WarpedCreateSpecialImageBatchFromVideo: Exception During Video File Read.")
        finally:
            cap.release()

        if len(last_image.shape) < 4:
            last_image = last_image.unsqueeze(0)

        while frame_count < num_frames:
            batched_images = torch.cat((batched_images, last_image.clone().detach()), 0)
            frame_count += 1

        print("WarpedCreateSpecialImageBatchFromVideo: Batched Images Shape: {}".format(batched_images.shape))

        return (batched_images, first_image, int(batched_images.shape[0]))

class WarpedBundleAllVideoImages:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"video_path": ("STRING", {"default": ""}),
                    "use_gpu": ("BOOLEAN", {"default": True}),
                    },
                }
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", "INT", "INT", "FLOAT", )
    RETURN_NAMES = ("image_batch", "first_image", "last_image", "num_frames", "width", "height", "fps", )
    FUNCTION = "generate"

    CATEGORY = "Warped/Video"

    def generate(self, video_path, use_gpu=False):
        if use_gpu:
            device = mm.get_torch_device()
        else:
            device = get_offload_device()

        print("WarpedBundleAllVideoImages: device: {}".format(device))

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('WarpedBundleAllVideoImages: width = %d' % width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('WarpedBundleAllVideoImages: height = %d' % height)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('WarpedBundleAllVideoImages: length = %d' % length)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = round(fps)
        print('WarpedBundleAllVideoImages: fps = %d' % fps)

        batched_images = None
        num_frames = 0
        last_image = None
        first_image = None

        try:
            while(cap.isOpened()):
                # Take each frame
                _, frameorig = cap.read()

                color_coverted = cv2.cvtColor(frameorig, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(color_coverted)
                temp_image = pil2tensorSwap(pil_image, device=device)

                if len(temp_image.shape) < 4:
                    temp_image = temp_image.unsqueeze(0)

                if first_image is None:
                    first_image = temp_image.clone().detach()

                last_image = temp_image.clone().detach()

                if not batched_images is None:
                    batched_images = torch.cat((batched_images, temp_image), 0)
                else:
                    batched_images = temp_image

                temp_image.to(get_offload_device())
                temp_image = None

                num_frames += 1

                if num_frames % 20 == 0:
                    print("WarpedBundleAllVideoImages: Frames Read: {}".format(num_frames))
        except:
            print("WarpedBundleAllVideoImages: Exception During Video File Read.")
        finally:
            cap.release()

        batched_images = batched_images.to(get_offload_device())

        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        return (batched_images, first_image, last_image, num_frames, width, height, float(fps), )

def augmentation_add_noise(image, noise_aug_strength, seed=None):
    if not seed is None:
        torch.manual_seed(seed)

    sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * noise_aug_strength
    image_noise = torch.randn_like(image) * sigma[:, None, None, None]
    image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
    image_out = image + image_noise
    return image_out

class WarpedImageNoiseAugmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_aug_strength": ("FLOAT", {"default": None, "min": 0.0, "max": 100.0, "step": 0.001}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", )
    RETURN_NAMES = ("image", "seed", )
    FUNCTION = "add_noise"

    CATEGORY = "Warped/Image"

    def add_noise(self, image, noise_aug_strength, seed):
        image_out = augmentation_add_noise(image, noise_aug_strength, seed)
        return image_out, seed,

def augmentation_patch_model(model, latent, index, strength, start_percent, end_percent):
    def outer_wrapper(samples, index, start_percent, end_percent):
        def unet_wrapper(apply_model, args):
            steps = args["c"]["transformer_options"]["sample_sigmas"]
            inp, timestep, c = args["input"], args["timestep"], args["c"]
            matched_step_index = (steps == timestep).nonzero()
            if len(matched_step_index) > 0:
                current_step_index = matched_step_index.item()
            else:
                for i in range(len(steps) - 1):
                    # walk from beginning of steps until crossing the timestep
                    if (steps[i] - timestep) * (steps[i + 1] - timestep) <= 0:
                        current_step_index = i
                        break
                else:
                    current_step_index = 0
            current_percent = current_step_index / (len(steps) - 1)
            if samples is not None:
                if start_percent <= current_percent <= end_percent:
                    inp[:, :, [index], :, :] = samples[:, :, [0], :, :].to(inp)
                else:
                    inp[:, :, [index], :, :] = torch.zeros(1)
            return apply_model(inp, timestep, **c)
        return unet_wrapper

    samples = latent["samples"] * 0.476986 * strength
    m = model.clone()
    m.set_model_unet_function_wrapper(outer_wrapper(samples, index, start_percent, end_percent))

    return m, 0.476986 * strength

class WarpedLeapfusionHunyuanI2V:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "index": ("INT", {"default": 0, "min": -1, "max": 1000, "step": 1,"tooltip": "The index of the latent to be replaced. 0 for first frame and -1 for last"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The start percentage of steps to apply"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The end percentage of steps to apply"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL", "FLOAT", )
    RETURN_NAMES = ("model", "scale_factor", )
    FUNCTION = "patch"

    CATEGORY = "Warped/HunyuanTools"

    def patch(self, model, latent, index, strength, start_percent, end_percent):
        new_model, scaling_strength = augmentation_patch_model(model, latent, index, strength, start_percent, end_percent)
        return (new_model, scaling_strength,)

class WarpedSaveAnimatedPng:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                     "png_filename": ("STRING", {"default": ""}),
                     "fps": ("FLOAT", {"default": 6.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                     "compress_level": ("INT", {"default": 4, "min": 0, "max": 9})
                     },
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Warped/Image/Animation"

    def save_images(self, images, fps, compress_level, png_filename=""):
        temp_filename = png_filename.split('.')

        i = 1
        file = temp_filename[0]
        while i < len(temp_filename) - 2:
            file = "{}.{}".format(file, temp_filename[i])
            i += 1

        file = "{}_{:05}_.webp".format(file, 1)

        filename_path = os.path.join(self.output_dir, png_filename)
        temp_path = os.path.join(self.temp_dir, file)
        filename = os.path.abspath(filename_path)

        print("Output Filename: {}".format(filename))

        preview_result = None

        results = list()
        pil_images = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if (img.width > 768) or (img.height > 768):
                img = self.scale_image(img.copy(), 768)

            pil_images.append(img)

        metadata = None
        num_frames = len(pil_images)
        print("Number Of Frames: {}".format(num_frames))

        pil_images[0].save(filename, pnginfo=metadata, compress_level=compress_level, save_all=True, duration=int(1000.0/fps), append_images=pil_images[1:num_frames])
        pil_images[0].save(temp_path, pnginfo=metadata, compress_level=compress_level, save_all=True, duration=int(1000.0/fps), append_images=pil_images[1:num_frames], lossless=False, quality=80, method=4)

        results.append({
            "filename": file,
            "subfolder": "",
            "type": "temp",
        })

        return { "ui": { "images": results, "animated": (True,)} }

    def scale_image(self, image, length=1024):
        img = image

        if img.height >= img.width:
            newHeight = length
            newWidth = int(float(length / img.height) * img.width)
        else:
            newWidth = length
            newHeight = int(float(length / img.width) * img.height)

        newImage = img.resize((newWidth, newHeight), resample=Image.BILINEAR)

        return newImage

class WarpedCFGGuider:
    def __init__(self):
        self.original_conds = {}
        self.cfg = 1.0
        self.inner_executor = None

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        self.cfg = cfg

    def inner_set_conds(self, conds):
        for k in conds:
            self.original_conds[k] = sampler_helpers.convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed):
        if latent_image is not None and torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed)

        extra_model_options = comfy_model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        extra_args = {"model_options": extra_model_options, "seed": seed}

        executor = patcher_extension.WrapperExecutor.new_class_executor(
            sampler.sample,
            sampler,
            patcher_extension.get_all_wrappers(patcher_extension.WrappersMP.SAMPLER_SAMPLE, extra_args["model_options"], is_model_options=True)
        )
        samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        return self.inner_model.process_latent_out(samples.to(torch.float32))

    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        self.inner_model, self.conds, self.loaded_models = sampler_helpers.prepare_sampling(self.model_patcher, noise.shape, self.conds, self.model_options)
        device = self.model_patcher.load_device

        if denoise_mask is not None:
            denoise_mask = sampler_helpers.prepare_mask(denoise_mask, noise.shape, device)

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)
        cast_to_load_options(self.model_options, device=device, dtype=self.model_patcher.model_dtype())

        try:
            self.model_patcher.pre_run()
            output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            self.model_patcher.cleanup()

        sampler_helpers.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.loaded_models
        return output

    def set_model(self, model):
        self.model_patcher = model
        self.model_options = model.model_options
        self.inner_executor = None

        return

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        if sigmas.shape[-1] == 0:
            return latent_image

        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
        preprocess_conds_hooks(self.conds)

        try:
            orig_model_options = self.model_options
            self.model_options = comfy_model_patcher.create_model_options_clone(self.model_options)
            # if one hook type (or just None), then don't bother caching weights for hooks (will never change after first step)
            orig_hook_mode = self.model_patcher.hook_mode
            if get_total_hook_groups_in_conds(self.conds) <= 1:
                self.model_patcher.hook_mode = hooks.EnumHookMode.MinVram
            sampler_helpers.prepare_model_patcher(self.model_patcher, self.conds, self.model_options)
            filter_registered_hooks_on_conds(self.conds, self.model_options)
            executor = patcher_extension.WrapperExecutor.new_class_executor(
                self.outer_sample,
                self,
                patcher_extension.get_all_wrappers(patcher_extension.WrappersMP.OUTER_SAMPLE, self.model_options, is_model_options=True)
            )
            output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)
            self.model_options = orig_model_options
            self.model_patcher.hook_mode = orig_hook_mode
            self.model_patcher.restore_hook_patches()

        del self.conds
        return output

class WarpedGuider_Basic(WarpedCFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})

class WarpedBasicGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"conditioning": ("CONDITIONING", ),
                    }
            }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "Warped/Sampling/Guiders"

    def get_guider(self, conditioning):
        guider = WarpedGuider_Basic()
        guider.set_conds(conditioning)
        return (guider,)

class WarpedUpscaleWithModel:
    rescale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def __init__(self):
        self.__imageScaler = ImageUpscaleWithModel()

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "images": ("IMAGE",),
                "upscale_by": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.05,
                }),
                "rescale_method": (self.rescale_methods,),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "upscale"

    CATEGORY = "Warped/Image"


    def upscale(self, upscale_model, images, upscale_by, rescale_method):
        upscaled_images = []

        samples = images.movedim(-1,1)

        width = round(samples.shape[3])
        height = round(samples.shape[2])

        target_width = round(samples.shape[3] * upscale_by)
        target_height = round(samples.shape[2] * upscale_by)


        samples = self.__imageScaler.upscale(upscale_model, images)[0].movedim(-1,1)

        upscaled_width = round(samples.shape[3])
        upscaled_height = round(samples.shape[2])

        if upscaled_width > target_width or upscaled_height > target_height:
            samples = comfy.utils.common_upscale(samples, target_width, target_height, rescale_method, "disabled")

        samples = samples.movedim(1,-1)

        return (samples,)

class WarpedLoadVideosBatch:
    def __init__(self):
        self.index = 0
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": '', "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", )
    RETURN_NAMES = ("mp4_filename", "png_filename", "webp_filename", )

    FUNCTION = "load_batch_videos"

    CATEGORY = "Warped/Video"

    def load_batch_videos(self, path):
        if not os.path.exists(path):
            return ("", "", )

        index=0
        mode="incremental_image"
        label='Batch 001'
        suffix=""

        retry = False

        try:
            return self.do_the_load(path, index, mode, label, suffix)
        except:
            self.index = 0
            retry = True

        if retry:
            return self.do_the_load(path, index, mode, label, suffix)

        return ("", "", )


    def do_the_load(self, path, index, mode, label, suffix):
        fl = self.BatchVideoLoader(path, label, '*', index)
        new_paths = fl.video_paths

        filename = fl.video_paths[self.index]

        tempStrings = filename.split('\\')
        png_filename = tempStrings[len(tempStrings) - 1]
        png_filename = png_filename.replace(".mp4", ".png")
        webp_filename = tempStrings[len(tempStrings) - 1]
        webp_filename = webp_filename.replace(".mp4", ".webp")

        print("Filename: {}".format(filename))
        print("Png Filename: {}".format(png_filename))
        print("Webp Filename: {}".format(webp_filename))

        self.index += 1

        if self.index >= len(fl.video_paths):
            self.index = 0

        return (filename, png_filename, webp_filename, )


    class BatchVideoLoader:
        def __init__(self, directory_path, label, pattern, index):
            self.video_paths = []
            self.load_videos(directory_path, pattern)
            self.video_paths.sort()

            self.index = index
            self.label = label

        def load_videos(self, directory_path, pattern):
            for file_name in glob.glob(os.path.join(directory_path, pattern), recursive=True):
                if file_name.lower().endswith(ALLOWED_VIDEO_EXT):
                    abs_file_path = os.path.abspath(file_name)
                    self.video_paths.append(abs_file_path)

        def get_video_by_id(self, video_id):
            if video_id < 0 or video_id >= len(self.video_paths):
                cstr(f"Invalid video index `{video_id}`").error.print()
                return

            return self.video_paths[video_id]

        def get_next_video(self):
            if self.index >= len(self.video_paths):
                self.index = 0

            video_path = self.video_paths[self.index]
            self.index += 1

            if self.index == len(self.video_paths):
                self.index = 0

            cstr(f'{cstr.color.YELLOW}{self.label}{cstr.color.END} Index: {self.index}').msg.print()

            return video_path

        def get_current_video(self):
            if self.index >= len(self.video_paths):
                self.index = 0
            video_path = self.video_paths[self.index]

            return video_path

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

class WarpedBundleVideoImages:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"video_path": ("STRING", {"default": ""}),
                    "starting_index": ("INT", {"default": 0}),
                    "num_frames": ("INT", {"default": 61, "min": 5, "max": 1000001, "step": 4}),
                    "use_gpu": ("BOOLEAN", {"default": True}),
                    },
                }
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", "INT", "INT", "FLOAT", )
    RETURN_NAMES = ("image_batch", "first_image", "last_image", "num_frames", "width", "height", "fps", )
    FUNCTION = "generate"

    CATEGORY = "Warped/Video"

    def generate(self, video_path, starting_index, num_frames, use_gpu=False):
        if use_gpu:
            device = mm.get_torch_device()
        else:
            device = get_offload_device()

        print("WarpedBundleVideoImages: device: {}".format(device))

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('WarpedBundleVideoImages: width = %d' % width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('WarpedBundleVideoImages: height = %d' % height)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('WarpedBundleVideoImages: length = %d' % length)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = round(fps)
        print('WarpedBundleVideoImages: fps = %d' % fps)

        batched_images = None
        # print_it = True
        last_image = None
        first_image = None

        if starting_index > length:
            starting_index = 0

        if ((starting_index + num_frames) > length) or (num_frames == 0):
            num_frames = length - starting_index

        try:
            if starting_index > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, starting_index)

            video_frames = []
            frame_count = 0
            for i in tqdm(range(num_frames), desc="Reading Video Frames: "):
                frame_count += 1

                # Take each frame
                _, frameorig = cap.read()

                color_coverted = cv2.cvtColor(frameorig, cv2.COLOR_BGR2RGB)
                video_frames.append(color_coverted)

            pil_image = Image.fromarray(video_frames[0])
            first_image = pil2tensorSwap(pil_image, device=device)
            first_image = first_image.to(get_offload_device())

            pil_image = Image.fromarray(video_frames[len(video_frames) - 1])
            last_image = pil2tensorSwap(pil_image, device=device)
            last_image = last_image.to(get_offload_device())

            for i in tqdm(range(len(video_frames)), desc="Preprocessing Video Frames: "):
                pil_image = Image.fromarray(video_frames[i])
                temp_image = pil2tensorSwap(pil_image, device=device)

                if len(temp_image.shape) < 4:
                    temp_image = temp_image.unsqueeze(0)

                if not batched_images is None:
                    batched_images = torch.cat((batched_images, temp_image), 0)
                else:
                    batched_images = temp_image

                temp_image.to(get_offload_device())
                temp_image = None
        except:
            print("WarpedBundleVideoImages: Exception During Video File Read.")
        finally:
            cap.release()

        batched_images = batched_images.to(get_offload_device())

        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        return (batched_images, first_image, last_image, int(batched_images.shape[0]), width, height, float(fps), )

class WarpedWanImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT", )
    RETURN_NAMES = ("positive", "negative", "latent", "num_frames", )
    FUNCTION = "encode"

    CATEGORY = "Warped/Video/Conditioning"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image = torch.mul(torch.ones((length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype), 0.5)
            image[:start_image.shape[0]] = start_image

            concat_latent_image = vae.encode(image[:, :, :, :3])

            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent, length, )

class WarpedImageFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                              "batch_index": ("INT", {"default": 0, "min": 0, "max": 4095}),
                              "length": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "frombatch"

    CATEGORY = "Warped/Image"

    def frombatch(self, image, batch_index, length):
        print("\n-------------------------------------------------------------------------------")
        print(image)
        print("-------------------------------------------------------------------------------\n")

        s_in = image

        if isinstance(s_in, list):
            s_in = s_in[0]

        batch_index = min(s_in.shape[0] - 1, batch_index)
        length = min(s_in.shape[0] - batch_index, length)
        s = s_in[batch_index:batch_index + length].clone()
        return (s,)

def teacache_hunyuanvideo_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        guiding_frame_index=None,
        control=None,
        transformer_options={},
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        coefficients = transformer_options.get("coefficients")
        max_skip_steps = transformer_options.get("max_skip_steps")

        initial_shape = list(img.shape)
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        if guiding_frame_index is not None:
            token_replace_vec = self.time_in(timestep_embedding(guiding_frame_index, 256, time_factor=1.0))
            vec_ = self.vector_in(y[:, :self.params.vec_in_dim])
            vec = torch.cat([(vec_ + token_replace_vec).unsqueeze(1), (vec_ + vec).unsqueeze(1)], dim=1)
            frame_tokens = (initial_shape[-1] // self.patch_size[-1]) * (initial_shape[-2] // self.patch_size[-2])
            modulation_dims = [(0, frame_tokens, 0), (frame_tokens, None, 1)]
            modulation_dims_txt = [(0, None, 1)]
        else:
            vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
            modulation_dims = None
            modulation_dims_txt = None

        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if txt_mask is not None and not torch.is_floating_point(txt_mask):
            txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

        txt = self.txt_in(txt, timesteps, txt_mask)

        ids = torch.cat((img_ids, txt_ids), dim=1)
        pe = self.pe_embedder(ids)

        img_len = img.shape[1]
        if txt_mask is not None:
            attn_mask_len = img_len + txt.shape[1]
            attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
            attn_mask[:, 0, img_len:] = txt_mask
        else:
            attn_mask = None

        blocks_replace = patches_replace.get("dit", {})

        # enable teacache
        img_mod1, _ = self.double_blocks[0].img_mod(vec)
        modulated_inp = self.double_blocks[0].img_norm1(img)
        modulated_inp = apply_mod(modulated_inp, (1 + img_mod1.scale), img_mod1.shift, modulation_dims)

        if not hasattr(self, 'accumulated_rel_l1_distance'):
            should_calc = True
            self.accumulated_rel_l1_distance = 0
            self.skip_steps = 0
        elif self.skip_steps == max_skip_steps:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
            self.skip_steps = 0
        else:
            try:
                self.accumulated_rel_l1_distance += poly1d(coefficients, ((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()))
                if self.accumulated_rel_l1_distance < rel_l1_thresh:
                    should_calc = False
                    self.skip_steps += 1
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
                    self.skip_steps = 0
            except:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
                self.skip_steps = 0

        self.previous_modulated_input = modulated_inp

        if not should_calc:
            img += self.previous_residual.to(img.device)
        else:
            ori_img = img.clone()
            for i, block in enumerate(self.double_blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims_img=args["modulation_dims_img"], modulation_dims_txt=args["modulation_dims_txt"])
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims_img': modulation_dims, 'modulation_dims_txt': modulation_dims_txt}, {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims_img=modulation_dims, modulation_dims_txt=modulation_dims_txt)

                if control is not None: # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

            img = torch.cat((img, txt), 1)

            for i, block in enumerate(self.single_blocks):
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims=args["modulation_dims"])
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims': modulation_dims}, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims=modulation_dims)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, : img_len] += add

            img = img[:, : img_len]
            self.previous_residual = (img - ori_img).to(mm.unet_offload_device())

        img = self.final_layer(img, vec, modulation_dims=modulation_dims)  # (N, T, patch_size ** 2 * out_channels)

        shape = initial_shape[-3:]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])
        return img

def teacache_wanmodel_forward(self, x, timestep, context, clip_fea=None, transformer_options={}, **kwargs):
        bs, c, t, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
        img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

        freqs = self.rope_embedder(img_ids).movedim(1, 2)
        return self.forward_orig(x, timestep, context, clip_fea, freqs, transformer_options)[:, :, :t, :h, :w]

def teacache_wanmodel_forward_orig(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options={},
    ):
        rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        coefficients = transformer_options.get("coefficients")
        max_skip_steps = transformer_options.get("max_skip_steps")
        cond_or_uncond = transformer_options.get("cond_or_uncond")

        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        if clip_fea is not None and self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            freqs=freqs,
            context=context)

        # enable teacache
        modulated_inp = e.to(mm.unet_offload_device())
        if not hasattr(self, 'teacache_state'):
            self.teacache_state = {
                0: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None, 'skip_steps': 0},
                1: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None, 'skip_steps': 0}
            }

        def update_cache_state(cache, modulated_inp):
            if cache['skip_steps'] == max_skip_steps:
                cache['should_calc'] = True
                cache['accumulated_rel_l1_distance'] = 0
                cache['skip_steps'] = 0
            elif cache['previous_modulated_input'] is not None:
                try:
                    if not self.teacache_state[0]["previous_residual"] is None:
                        cache['accumulated_rel_l1_distance'] += poly1d(coefficients, ((modulated_inp-cache['previous_modulated_input']).abs().mean() / cache['previous_modulated_input'].abs().mean()))
                    else:
                        cache['accumulated_rel_l1_distance'] += poly1d(INITIAL_COEFFICIENTS, ((modulated_inp-cache['previous_modulated_input']).abs().mean() / cache['previous_modulated_input'].abs().mean()))

                    if cache['accumulated_rel_l1_distance'] < rel_l1_thresh:
                        cache['should_calc'] = False
                        cache['skip_steps'] += 1
                    else:
                        cache['should_calc'] = True
                        cache['accumulated_rel_l1_distance'] = 0
                        cache['skip_steps'] = 0
                except:
                    cache['should_calc'] = True
                    cache['accumulated_rel_l1_distance'] = 0
                    cache['skip_steps'] = 0
            cache['previous_modulated_input'] = modulated_inp

        b = int(len(x) / len(cond_or_uncond))

        for i, k in enumerate(cond_or_uncond):
            update_cache_state(self.teacache_state[k], modulated_inp[i*b:(i+1)*b])

        should_calc = False
        for k in cond_or_uncond:
            should_calc = (should_calc or self.teacache_state[k]['should_calc'])

        if not should_calc:
            for i, k in enumerate(cond_or_uncond):
                x[i*b:(i+1)*b] += self.teacache_state[k]['previous_residual'].to(x.device)
        else:
            ori_x = x.clone()
            for block in self.blocks:
                x = block(x, **kwargs)
            for i, k in enumerate(cond_or_uncond):
                self.teacache_state[k]['previous_residual'] = (x - ori_x)[i*b:(i+1)*b].to(mm.unet_offload_device())

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

def clear_teacache_state(self):
    self.teacache_state = {
        0: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None, 'skip_steps': 0},
        1: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None, 'skip_steps': 0}
    }

class WarpedTeaCache:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the TeaCache will be applied to."}),
                "model_type": (["wan2.1_t2v_1.3B", "wan2.1_t2v_14B", "wan2.1_i2v_480p_14B", "wan2.1_i2v_720p_14B", "hunyuan_video"], {"default": "wan2.1_i2v_720p_14B", "tooltip": "Supported diffusion model."}),
                "rel_l1_thresh": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."}),
                "max_skip_steps": ([1, 2, 3], {"default": 3, "tooltip": "Max continuous skip steps."})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_teacache"
    CATEGORY = "Warped/TeaCache"
    TITLE = "Warped TeaCache"

    def apply_teacache(self, model, model_type: str, rel_l1_thresh: float, max_skip_steps: int):
        if rel_l1_thresh == 0:
            return (model,)

        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}
        new_model.model_options["transformer_options"]["rel_l1_thresh"] = rel_l1_thresh
        new_model.model_options["transformer_options"]["max_skip_steps"] = max_skip_steps
        new_model.model_options["transformer_options"]["coefficients"] = SUPPORTED_MODELS_COEFFICIENTS[model_type]
        diffusion_model = new_model.get_model_object("diffusion_model")

        if "hunyuan_video" in model_type:
            is_cfg = False
            context = patch.multiple(
                diffusion_model,
                forward_orig=teacache_hunyuanvideo_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "wan2.1" in model_type:
            is_cfg = True
            context = patch.multiple(
                diffusion_model,
                forward=teacache_wanmodel_forward.__get__(diffusion_model, diffusion_model.__class__),
                forward_orig=teacache_wanmodel_forward_orig.__get__(diffusion_model, diffusion_model.__class__)
            )
        else:
            raise ValueError(f"Unknown type {model_type}")

        if hasattr(diffusion_model, 'teacache_state'):
            delattr(diffusion_model, 'teacache_state')

        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            cond_or_uncond = kwargs["cond_or_uncond"]
            # referenced from https://github.com/kijai/ComfyUI-KJNodes/blob/d126b62cebee81ea14ec06ea7cd7526999cb0554/nodes/model_optimization_nodes.py#L868
            sigmas = c["transformer_options"]["sample_sigmas"]
            matched_step_index = (sigmas == timestep[0]).nonzero()
            if len(matched_step_index) > 0:
                current_step_index = matched_step_index.item()
            else:
                current_step_index = 0
                for i in range(len(sigmas) - 1):
                    # walk from beginning of steps until crossing the timestep
                    if (sigmas[i] - timestep) * (sigmas[i + 1] - timestep) <= 0:
                        current_step_index = i
                        break

            if current_step_index == 0:
                if is_cfg:
                    # uncond first
                    if (1 in cond_or_uncond) and hasattr(diffusion_model, 'teacache_state'):
                        delattr(diffusion_model, 'teacache_state')
                else:
                    if hasattr(diffusion_model, 'accumulated_rel_l1_distance'):
                        delattr(diffusion_model, 'accumulated_rel_l1_distance')

            with context:
                return model_function(input, timestep, **c)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)

        return (new_model,)

class WarpedBaseWanLoraLoader:
    """Base class for Wan LoRA loading functionality"""

    def __init__(self):
        self.loaded_lora: Optional[Tuple[str, Dict[str, torch.Tensor]]] = None

    @classmethod
    def get_cache_dir(cls) -> str:
        """Get or create the cache directory for block settings"""
        try:
            from folder_paths import base_path, folder_names_and_paths
            cache_dir = Path(folder_names_and_paths["custom_nodes"][0][0]) / "ComfyUI-WarpedToolset" / "cache"
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            return cache_dir
        except Exception as e:
            logger.error(f"Failed to create or access cache directory: {str(e)}")
            raise

    def get_settings_filename(self, lora_name: str) -> str:
        """Generate the settings filename for a given LoRA"""
        base_name = os.path.splitext(lora_name)[0]
        return os.path.join(self.get_cache_dir(), f"{base_name}_blocks.yaml")

    def get_block_settings(self, lora_name: str, use_block_cache: bool = True, include_single_blocks: bool = False) -> dict:
        """Load block settings from cache or return defaults"""
        # Initialize with all double blocks enabled and single blocks based on parameter
        default_settings = {
            **{f"blocks.{i}.": True for i in range(40)},
        }

        if not use_block_cache:
            return default_settings

        try:
            settings_file = self.get_settings_filename(lora_name)
            if os.path.exists(settings_file):
                cached_settings = yaml.safe_load(open(settings_file, 'r'))
                # Merge cached settings with default single block settings
                return {
                    **default_settings,
                    **cached_settings,
                }
            return default_settings
        except Exception as e:
            logger.error(f"Failed to load block settings for {lora_name}: {str(e)}")
            return default_settings

    def save_block_settings(self, lora_name: str, block_settings: dict):
        """Save block settings to cache"""
        try:
            settings_file = self.get_settings_filename(lora_name)
            # Ensure directory exists
            os.makedirs(os.path.dirname(settings_file), exist_ok=True)
            save_settings = {k: v for k, v in block_settings.items() if k.startswith('blocks.')}
            with open(settings_file, 'w') as f:
                yaml.safe_dump(save_settings, f)
        except Exception as e:
            logger.error(f"Failed to save block settings for {lora_name}: {str(e)}")

    def filter_lora_keys(self, lora: Dict[str, torch.Tensor], block_settings: dict) -> Dict[str, torch.Tensor]:
        """Filter LoRA keys based on block settings"""
        filtered_blocks = {k: v for k, v in block_settings.items() if v is True}
        return {key: value for key, value in lora.items()
                if any(block in key for block in filtered_blocks)}

    def load_lora_file(self, lora_name: str) -> Dict[str, torch.Tensor]:
        """Load LoRA file and cache it"""
        from comfy.utils import load_torch_file

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise Exception(f"LoRA {lora_name} not found at {lora_path}")

        if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
            return self.loaded_lora[1]

        lora = load_torch_file(lora_path)
        self.loaded_lora = (lora_path, lora)
        return lora

    def get_file_mtime(self, filepath: str) -> str:
        """Get modification time of file as string"""
        try:
            return str(os.path.getmtime(filepath))
        except:
            return "0"

    def get_lora_mtime(self, lora_name: str) -> str:
        """Get modification time of LoRA file"""
        try:
            lora_path = folder_paths.get_full_path("loras", lora_name)
            return self.get_file_mtime(lora_path)
        except:
            return "0"

    def get_cache_mtime(self, lora_name: str) -> str:
        """Get modification time of cache file"""
        try:
            cache_file = self.get_settings_filename(lora_name)
            return self.get_file_mtime(cache_file)
        except:
            return "0"

    def apply_lora(self, model, clip, lora_name: str, strength_model: float, strength_clip: float, block_settings: Optional[dict] = None) -> torch.nn.Module:
        """Apply LoRA to model with given settings"""
        from comfy.sd import load_lora_for_models

        if not lora_name:
            return model

        try:
            lora = self.load_lora_file(lora_name)
            if block_settings is None:
                block_settings = self.get_block_settings(lora_name, True)  # Always use cache for direct loading

            filtered_lora = self.filter_lora_keys(lora, block_settings)
            new_model, new_clip = load_lora_for_models(model, clip, filtered_lora, strength_model, strength_clip)

            # print("\n=============================================================")
            # print("Normal")
            # print(new_model)
            # print("=============================================================\n")
            #
            # print("\n=============================================================")
            # print(new_clip)
            # print("=============================================================\n")

            return (new_model, new_clip,)

        except Exception as e:
            logger.error(f"Error applying LoRA {lora_name}: {str(e)}")

            # print("\n=============================================================")
            # print("Exception")
            # print(model)
            # print("=============================================================\n")
            #
            # print("\n=============================================================")
            # print(clip)
            # print("=============================================================\n")

            return (model, clip, )

class WarpedWanLoadAndEditLoraBlocks(WarpedBaseWanLoraLoader):
    """Interactive LoRA block editor"""

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
                "save_settings": ("BOOLEAN", {"default": True}),
            }
        }

        for i in range(40):
            arg_dict["required"][f"blocks.{i}."] = ("BOOLEAN", {"default": True})

        return arg_dict

    RETURN_TYPES = ("MODEL", "CLIP", )
    RETURN_NAMES = ("model", "clip", )
    FUNCTION = "load_lora"
    CATEGORY = "Warped/WanTools"

    @classmethod
    def IS_CHANGED(s, model, clip, lora_name: str, strength_model: float, strength_clip: float, save_settings: bool, **kwargs):
        instance = s()
        lora_mtime = instance.get_lora_mtime(lora_name)
        return f"{lora_name}_{strength_model}_{strength_clip}_{lora_mtime}"

    def load_lora(self, model, clip, lora_name: str, strength_model: float, strength_clip: float, save_settings: bool, **kwargs):
        if not lora_name:
            return (model,)

        # Add single blocks settings based on the parameter
        block_settings = {
            **kwargs,
        }

        if save_settings:
            self.save_block_settings(lora_name, block_settings)

        return_model, return_clip = self.apply_lora(model, clip, lora_name, strength_model, strength_clip, block_settings)

        return (return_model, return_clip,)

class WarpedHunyuanImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE", ),
                    "positive": ("CONDITIONING", ),
                    "vae": ("VAE", ),
                    "num_frames": ("INT", {"default": 53, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                    "guidance_type": (["v1 (concat)", "v2 (replace)"], )
                    },
                }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"

    CATEGORY = "Warped/Video/Conditioning"

    def encode(self, images, positive, vae, num_frames, guidance_type):
        images.to(dtype=torch.float32, device=get_offload_device())

        if len(images.shape) < 4:
            images = images.unsqueeze()

        height = images.shape[1]
        width  = images.shape[2]

        out_latent = {}

        if images.shape[0] == 1:
            print("Single Image")
            latent = torch.zeros([1, 16, ((num_frames - 1) // 4) + 1, height // 8, width // 8], dtype=torch.float32, device=get_offload_device())

            images = comfy.utils.common_upscale(images[:num_frames, :, :, :3].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

            concat_latent_image = vae.encode(images)
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=images.device, dtype=images.dtype)
            mask[:, :, :((images.shape[0] - 1) // 4) + 1] = 0.0

            if guidance_type == "v1 (concat)":
                cond = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            else:
                cond = {'guiding_frame_index': 0}
                latent[:, :, :concat_latent_image.shape[2]] = concat_latent_image
                out_latent["noise_mask"] = mask

            positive = node_helpers.conditioning_set_values(positive, cond)

            out_latent["samples"] = latent

            return (positive, out_latent)

        print("Shape Before: {}".format(images.shape))

        images = comfy.utils.common_upscale(images.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        # images = torch.mul(torch.ones((num_frames, height, width, images.shape[-1]), device=images.device, dtype=images.dtype), 0.5)

        print("Shape After: {}".format(images.shape))

        concat_latent_image = partial_encode_tiled(vae, images)

        if len(concat_latent_image.shape) < 5:
            concat_latent_image = concat_latent_image.unsqueeze(0)

        mask = torch.ones((1, 1, concat_latent_image.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=images.device, dtype=images.dtype)
        # mask[:, :, :((images.shape[0] - 1) // 4) + 1] = 0.0

        out_latent["samples"] = concat_latent_image

        if guidance_type == "v1 (concat)":
            cond = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        else:
            cond = {'guiding_frame_index': 0}
            # out_latent["noise_mask"] = mask

        positive = node_helpers.conditioning_set_values(positive, cond)

        return (positive, out_latent)

class WarpedImageResizeKJStyle:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, }),
                "upscale_method": (s.upscale_methods,),
                "keep_proportion": ("BOOLEAN", { "default": False }),
                "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
            },
            "optional" : {
                "width_input": ("INT", { "forceInput": True}),
                "height_input": ("INT", { "forceInput": True}),
                "get_image_size": ("IMAGE",),
                "crop": (["disabled","center"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = "Warped/Image"
    DESCRIPTION = """
Resizes the image to the specified width and height.
Size can be retrieved from the inputs, and the final scale
is  determined in this order of importance:
- get_image_size
- width_input and height_input
- width and height widgets

Keep proportions keeps the aspect ratio of the image, by
highest dimension.
"""

    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by,
               width_input=None, height_input=None, get_image_size=None, crop="disabled"):
        B, H, W, C = image.shape

        if width_input:
            width = width_input
        if height_input:
            height = height_input
        if get_image_size is not None:
            _, height, width, _ = get_image_size.shape

        if keep_proportion and get_image_size is None:
                # If one of the dimensions is zero, calculate it to maintain the aspect ratio
                if width == 0 and height != 0:
                    ratio = height / H
                    width = round(W * ratio)
                elif height == 0 and width != 0:
                    ratio = width / W
                    height = round(H * ratio)
                elif width != 0 and height != 0:
                    # Scale based on which dimension is smaller in proportion to the desired dimensions
                    ratio = min(width / W, height / H)
                    width = round(W * ratio)
                    height = round(H * ratio)
        else:
            if width == 0:
                width = W
            if height == 0:
                height = H

        if divisible_by > 1 and get_image_size is None:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        image = image.movedim(-1,1)
        image = utils.common_upscale(image, width, height, upscale_method, crop)
        image = image.movedim(1,-1)

        return(image, image.shape[2], image.shape[1],)

class WarpedImageResize:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 400, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 720, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, }),
                "upscale_method": (self.upscale_methods, {"default": "lanczos"}),
                "crop": (["center", "top_left", "top_right", "bottom_left", "bottom_right", "top_center", "bottom_center"], {"default": "center"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "scale_orig_image", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = "Warped/Image"
    DESCRIPTION = """
                Resizes the image to the specified width and height.
                Size can be retrieved from the inputs, and the final scale
                is  determined in this order of importance:
                - get_image_size
                - width_input and height_input
                - width and height widgets

                Keep proportions keeps the aspect ratio of the image, by
                highest dimension.
                """

    def resize(self, image, width, height, upscale_method, crop="center"):
        print("Original Image Shape: {}".format(image.shape))

        B, H, W, C = image.shape
        self.is_downsize = False

        scaled_image = image.clone().detach()
        test_image1  = image.clone().detach()
        test_image2  = image.clone().detach()

        is_long_side, orig_is_landscape, new_is_landscape, upscale_required = self.determine_side_to_scale(H, W, width, height)

        print("Automatic Value Determination: is_long_side: {}  |  original_is_landscape: {}  |  new_is_landscape: {}  | upscale_required {}".format(is_long_side, orig_is_landscape, new_is_landscape, upscale_required))

        if upscale_required:
            W = int(W / 16) * 16
            H = int(H / 16) * 16

            print("Modified Width: {}  |  Height: {}".format(W, H))

            # Scale based on which dimension is smaller in proportion to the desired dimensions
            ratio = max(width / W, height / H)
            temp_width = round(W * ratio)
            temp_height = round(H * ratio)

            print("Aspect Ratio Change Required: temp_width: {}  |  temp_height: {}".format(temp_width, temp_height))

            image = image.movedim(-1,1)
            image = self.upscale(image, temp_width, temp_height, upscale_method, crop)
            image = image.movedim(1,-1)

            if orig_is_landscape:
                if is_long_side:
                    image = self.scale_to_side(image, width, is_long_side)
                else:
                    image = self.scale_to_side(image, height, is_long_side)
            else:
                if is_long_side:
                    image = self.scale_to_side(image, height, is_long_side)
                else:
                    image = self.scale_to_side(image, width, is_long_side)

            scaled_image = image.clone().detach()

            image = self.crop(image, width, height, crop)

            return(image, scaled_image, image.shape[2], image.shape[1],)

        if (width < W) or (height < H):
            # if same orientation
            if (orig_is_landscape and new_is_landscape) or (not orig_is_landscape and not new_is_landscape):
                if is_long_side:
                    if not new_is_landscape:
                        image = self.scale_to_side(image, height, is_long_side)
                    else:
                        image = self.scale_to_side(image, width, is_long_side)
                else:
                    if new_is_landscape:
                        image = self.scale_to_side(image, height, is_long_side)
                    else:
                        image = self.scale_to_side(image, width, is_long_side)
            # if original is landscape and new is portrait or original is portrait and new is landscape
            else:
                if is_long_side:
                    image = self.scale_to_side(image, width, is_long_side)
                else:
                    image = self.scale_to_side(image, height, is_long_side)

            scaled_image = image.clone().detach()

            B, H, W, C = image.shape
            self.is_downsize = True

        if self.is_downsize:
            image = self.crop(image, width, height, crop)
            return(image, scaled_image, image.shape[2], image.shape[1],)

        if (orig_is_landscape and new_is_landscape) or ((not orig_is_landscape) and (not new_is_landscape)):
            image = image.movedim(-1,1)
            image = self.upscale(image, width, height, upscale_method, crop)
            image = image.movedim(1,-1)

            return(image, image, image.shape[2], image.shape[1],)

        if orig_is_landscape:
            temp_ratio = round(height // H)
            temp_width = round(W * temp_ratio)

            image = image.movedim(-1,1)
            image = self.upscale(image, temp_width, height, upscale_method, "disabled")
            image = image.movedim(1,-1)

            scaled_image = image.clone().detach()
            image = self.crop(image, width, height, crop)

            return(image, scaled_image, image.shape[2], image.shape[1],)

        temp_ratio = round(width // W)
        temp_height = round(H * temp_ratio)

        image = image.movedim(-1,1)
        image = self.upscale(image, width, temp_height, upscale_method, "disabled")
        image = image.movedim(1,-1)

        scaled_image = image.clone().detach()
        image = self.crop(image, width, height, crop)

        return(image, scaled_image, image.shape[2], image.shape[1],)

    def determine_side_to_scale(self, original_height, original_width, width, height):
        original_is_landscape = False
        new_is_landscape = False

        if original_width > original_height:
            original_is_landscape = True

        if width > height:
            new_is_landscape = True

        if (original_width == original_height):
            if width <= height:
                original_is_landscape = True

                if width < height:
                    new_is_landscape = True
                else:
                    new_is_landscape = original_is_landscape
            else:
                original_is_landscape = True
                new_is_landscape = True

        is_long_side = True
        upscale_required = False

        if (not new_is_landscape and original_is_landscape) or (new_is_landscape and not original_is_landscape):
            is_long_side = False
        else:
            if original_is_landscape:
                temp_ratio  = round(width // original_width)
                temp_height = round(temp_ratio * height)

                if temp_height < height:
                    is_long_side = False
                    upscale_required = True
            else:
                temp_ratio  = round(height // original_height)
                temp_width = round(temp_ratio * width)

                if temp_width < width:
                    is_long_side = False
                    upscale_required = True

        return is_long_side, original_is_landscape, new_is_landscape, upscale_required

    def crop(self, image, width, height, crop_type):
        new_image = tensor2pilSwap(image)
        new_image = new_image[0]

        print("Image width: {} height: {}  |  New width: {} height: {}".format(new_image.width, new_image.height, width, height))

        #(left, upper, right, lower)
        if crop_type == "top_left":
            left = 0
            upper = 0
            right = width
            lower = height
        elif crop_type == "top_right":
            left = new_image.width - width
            upper = 0
            right = new_image.width
            lower = height
        elif crop_type == "top_center":
            left = int(new_image.width // 2) - int(width // 2)
            upper = 0
            right = (int(new_image.width // 2) - int(width // 2)) + width
            lower = height
        elif crop_type == "bottom_left":
            left = 0
            upper = new_image.height - height
            right = width
            lower = new_image.height
        elif crop_type == "bottom_right":
            left = new_image.width - width
            upper = new_image.height - height
            right = new_image.width
            lower = new_image.height
        elif crop_type == "bottom_center":
            left = int(new_image.width // 2) - int(width // 2)
            upper = new_image.height - height
            right = (int(new_image.width // 2) - int(width // 2)) + width
            lower = new_image.height
        elif crop_type == "center":
            left = int(new_image.width // 2) - int(width // 2)
            upper = int(new_image.height // 2) - int(height // 2)
            right = (int(new_image.width // 2) - int(width // 2)) + width
            lower = (int(new_image.height // 2) - int(height // 2)) + height

        print("Crop Locations: Left: {}  |  Upper: {}  |  Right: {}  |  Lower: {}".format(int(new_image.width // 2) - int(width // 2), 0, width, height))
        new_image = new_image.crop((left, upper, right, lower))

        new_image = pil2tensorSwap(new_image)

        return new_image

    def upscale(self, samples, width, height, upscale_method, crop):
            orig_shape = tuple(samples.shape)
            if len(orig_shape) > 4:
                samples = samples.reshape(samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
                samples = samples.movedim(2, 1)
                samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])

            s = samples
            if crop != "disabed":
                old_width = samples.shape[-1]
                old_height = samples.shape[-2]
                old_aspect = old_width / old_height
                new_aspect = width / height
                x = 0
                y = 0

                # ["disabled","center", "top_left", "top_right", "bottom_left", "bottom_right", "top_center", "bottom_center"]
                # narrow(dim, start, length)
                if crop == "center":
                    if old_aspect > new_aspect:
                        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
                    elif old_aspect < new_aspect:
                        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)

                    s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
                elif (s.shape[2] != height) or (s.shape[3] != width):
                    if self.is_downsize:
                        if len(s.shape) < 4:
                            s = s.unsqueeze(0)

                        print("s.shape: {}".format(s.shape))

                        if crop == "top_left":
                            if width < height:
                                s = s.narrow(-1, 0, width)
                            else:
                                s = s.narrow(-2, 0, height)
                        elif crop == "top_right":
                            if width < height:
                                s = s.narrow(-1, s.shape[3] - width, width)
                            else:
                                s = s.narrow(-2, 0, height)
                        elif crop == "bottom_left":
                            if width < height:
                                s = s.narrow(-1, 0, width)
                            else:
                                s = s.narrow(-2, s.shape[2] - height, height)
                        elif crop == "bottom_right":
                            if width < height:
                                s = s.narrow(-1, s.shape[3] -  width, width)
                            else:
                                s = s.narrow(-2, s.shape[2] - height, height)
                        elif crop == "top_center":
                            if width < height:
                                s = s.narrow(-1, round((s.shape[3] / 2) - (width / 2) - 1), width)
                            else:
                                s = s.narrow(-1, 0, height)
                        elif crop == "bottom_center":
                            if width < height:
                                s = s.narrow(-1, round((s.shape[3] / 2) - (width / 2) - 1), width)
                                print("s.shape 1: {}".format(s.shape))
                            else:
                                s = s.narrow(-2, s.shape[2] - height, height)
                                print("s.shape 2: {}".format(s.shape))

            if upscale_method == "bislerp":
                out = utils.bislerp(s, width, height)
            elif upscale_method == "lanczos":
                out = utils.lanczos(s, width, height)
            else:
                out = torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

            if len(orig_shape) == 4:
                return out

            out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
            return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))

    def scale_to_side(self, image, length=1024, scale_long_side=True):
        img = tensor2pilSwap(image)
        img = img[0]

        if scale_long_side:
            if img.height >= img.width:
                newHeight = length
                newWidth = int(float(length / img.height) * img.width)
            else:
                newWidth = length
                newHeight = int(float(length / img.width) * img.height)
        else:
            if img.height <= img.width:
                newHeight = length
                newWidth = int(float(length / img.height) * img.width)
            else:
                newWidth = length
                newHeight = int(float(length / img.width) * img.height)

        tempImage = img.resize((newWidth, newHeight), resample=Image.BILINEAR)

        newImage = pil2tensorSwap([tempImage])

        return newImage

class WarpedImageScaleToSide:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE", ),
                     "length": ("INT", {"default": 1024}),
                     "scale_to": (["long_side", "short_side"], {"default": "long_side"}),
                     "use_gpu": ("BOOLEAN", {"default": False}),
                    },
                }

    CATEGORY = "Warped/Image"

    RETURN_TYPES = ("IMAGE", "INT", "INT", )
    RETURN_NAMES = ("image", "width", "height")

    FUNCTION = "scale_image"

    def scale_image(self, image, length=1024, scale_to="long_side", use_gpu=False):
        if use_gpu:
            device = mm.get_torch_device()
        else:
            device = get_offload_device()

        print("WarpedImageScaleToSide: device: {}".format(device))

        if len(image.shape) < 4:
            image = image.unsqueeze(0)

        image_batches_tuple = torch.split(image, 1, dim=0)
        image_batches_split = [item for item in image_batches_tuple]

        final_image = None

        # for i in tqdm(range(num_frames), desc="Preprocessing Video Frames: "):
        for single_image in image_batches_split:
            img = tensor2pilSwap(single_image)
            img = img[0]

            if scale_to == "long_side":
                if img.height >= img.width:
                    newHeight = length
                    newWidth = int(float(length / img.height) * img.width)
                else:
                    newWidth = length
                    newHeight = int(float(length / img.width) * img.height)
            else:
                if img.height <= img.width:
                    newHeight = length
                    newWidth = int(float(length / img.height) * img.width)
                else:
                    newWidth = length
                    newHeight = int(float(length / img.width) * img.height)

            tempImage = img.resize((newWidth, newHeight), resample=Image.BILINEAR)

            newImage = pil2tensorSwap([tempImage], device=device)

            if len(newImage.shape) < 4:
                newImage = newImage.unsqueeze(0)

            if not final_image is None:
                final_image = torch.cat((final_image, newImage), 0)
            else:
                final_image = newImage

        newImage.to(get_offload_device())
        newImage = None
        final_image.to(get_offload_device())

        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        return (final_image, final_image.shape[1], final_image.shape[2],)

class WarpedHunyuanLoraCheck:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": (['None'] + get_lora_list(),),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "check_lora"
    CATEGORY = "Warped/HunyuanTools"
    DESCRIPTION = ""

    def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load and filter a single LoRA model."""
        if not lora_name or strength == 0:
            return {}, {}

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        # Load the LoRA weights
        lora_weights = utils.load_torch_file(lora_path)

        return lora_weights

    def check_lora(self, lora):
        if lora != "None":
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_1, 1.0, blocks_type_1)
        else:
            lora_weights = {}

        for key in lora_weights:
            print(key)

        return {"ui": {"tags": [save_message]}}

def get_base_lora_dirs():
    return folder_paths.get_folder_paths("loras")

# class WarpedLoadLorasBatch:
#     def __init__(self):
#         self.index = 0
#         self.base_lora_dir = ""
#         self.sub_folder = ""
#
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "base_lora_dir": (get_base_lora_dirs(), ),
#                 "lora_subdirectory": ("STRING", {"default": '', "multiline": False}),
#             },
#         }
#
#     RETURN_TYPES = ("STRING", "STRING", )
#     RETURN_NAMES = ("lora_name", "full_lora_path", )
#     FUNCTION = "load_batch_loras"
#
#     CATEGORY = "Warped/LORA"
#
#     def load_batch_loras(self, base_lora_dir, lora_subdirectory):
#         self.sub_folder = lora_subdirectory
#         self.base_lora_dir = base_lora_dir
#         path = os.path.join(self.base_lora_dir, self.sub_folder)
#         print(path)
#
#         if not os.path.exists(path):
#             return ("", "", )
#
#         index=0
#         mode="incremental_lora"
#         label='Batch 001'
#         suffix=""
#
#         retry = False
#
#         try:
#             filename, full_filename = self.do_the_load(path, index, mode, label, suffix)
#             print("WarpedLoadLorasBatch: Filename: {}  |  Full File Path: {}".format(filename, full_filename))
#             return (filename, full_filename, )
#         except:
#             self.index = 0
#             retry = True
#
#         if retry:
#             filename, full_filename = self.do_the_load(path, index, mode, label, suffix)
#             print("WarpedLoadLorasBatch: Retrying: Filename: {}  |  Full File Path: {}".format(filename, full_filename))
#             return (filename, full_filename, )
#
#         return ("", "", )
#
#
#     def do_the_load(self, path, index, mode, label, suffix):
#         fl = self.BatchLoraLoader(path, label, '*', index)
#         new_paths = fl.lora_paths
#
#         filename = fl.lora_paths[self.index]
#
#         filename = os.path.join(self.sub_folder, filename)
#         full_filename = os.path.join(self.base_lora_dir, filename)
#
#         self.index += 1
#
#         if self.index >= len(fl.lora_paths):
#             self.index = 0
#
#         return filename, full_filename
#
#
#     class BatchLoraLoader:
#         def __init__(self, directory_path, label, pattern, index):
#             self.lora_paths = []
#             self.load_loras(directory_path, pattern)
#             self.lora_paths.sort()
#
#             self.index = index
#             self.label = label
#
#         def load_loras(self, directory_path, pattern):
#             for file_name in glob.glob(os.path.join(directory_path, pattern), recursive=True):
#                 temp_strings = file_name.split('\\')
#                 file_name = temp_strings[len(temp_strings) - 1]
#
#                 if file_name.lower().endswith("safetensors"):
#                     # abs_file_path = os.path.abspath(file_name)
#                     self.lora_paths.append(file_name)
#
#         def get_lora_by_id(self, lora_id):
#             if lora_id < 0 or lora_id >= len(self.lora_paths):
#                 cstr(f"Invalid lora index `{lora_id}`").error.print()
#                 return
#
#             return self.lora_paths[lora_id]
#
#         def get_next_lora(self):
#             if self.index >= len(self.lora_paths):
#                 self.index = 0
#
#             lora_path = self.lora_paths[self.index]
#             self.index += 1
#
#             if self.index == len(self.lora_paths):
#                 self.index = 0
#
#             cstr(f'{cstr.color.YELLOW}{self.label}{cstr.color.END} Index: {self.index}').msg.print()
#
#             return lora_path
#
#         def get_current_lora(self):
#             if self.index >= len(self.lora_paths):
#                 self.index = 0
#             lora_path = self.lora_paths[self.index]
#
#             return lora_path
#
#     @classmethod
#     def IS_CHANGED(cls, **kwargs):
#         return float("NaN")

def get_lora_directories():
    lora_dirs = get_base_lora_dirs()

    result_lora_dirs = []

    for lora_dir in lora_dirs:
        temp_dirs = [x[0] for x in os.walk(lora_dir)]
        result_lora_dirs = result_lora_dirs + temp_dirs

    result_lora_dirs.sort()

    return(result_lora_dirs)

def get_lora_path_parts(path):
    temp_base_dirs = get_base_lora_dirs()
    base_dir = ""
    lora_name = ""

    for temp_dir in temp_base_dirs:
        if path.startswith(temp_dir):
            base_dir = temp_dir
            lora_name = path[len(base_dir) + 1:]
            print("get_lora_path_parts: base_dir: {}  |  lora_name: {}".format(base_dir, lora_name))
            break

    return base_dir, lora_name

class WarpedLoadLorasBatchByPrefix:
    def __init__(self):
        self.index = 0
        self.lora_dir = ""
        self.last_prefix = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_dir": (get_lora_directories(), ),
                "lora_prefix": ("STRING", {"default": '', "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", )
    RETURN_NAMES = ("lora_name", "full_lora_path", )
    FUNCTION = "load_batch_loras"

    CATEGORY = "Warped/LORA"

    def load_batch_loras(self, lora_dir, lora_prefix):
        self.lora_dir = lora_dir
        path = lora_dir
        print(path)

        if not os.path.exists(path):
            return ("", "", )

        if not (self.last_prefix == lora_prefix):
            self.last_prefix = lora_prefix
            self.index = 0

        retry = False
        index = 0

        try:
            filename, full_filename = self.do_the_load(path, lora_prefix, index)
            print("WarpedLoadLorasBatchByPrefix: Filename: {}  |  Full File Path: {}".format(filename, full_filename))
            return (filename, full_filename, )
        except:
            self.index = 0
            retry = True

        if retry:
            retry = False
            filename, full_filename = self.do_the_load(path, lora_prefix, index)
            print("WarpedLoadLorasBatchByPrefix: Retrying: Filename: {}  |  Full File Path: {}".format(filename, full_filename))
            return (filename, full_filename, )

        return ("", "", )


    def do_the_load(self, path, prefix, index):
        prefix = prefix.strip(' ')

        if (len(prefix) == 1) and (prefix == '*'):
            fl = self.BatchLoraLoader(path, '*', index)
        else:
            prefix = prefix.strip('*')
            fl = self.BatchLoraLoader(path, "{}*".format(prefix), index)

        new_paths = fl.lora_paths

        filename = fl.lora_paths[self.index]

        # filename = os.path.join(self.sub_folder, filename)
        full_filename = os.path.join(path, filename)
        base_dir, lora_name = get_lora_path_parts(full_filename)

        self.index += 1

        if self.index >= len(fl.lora_paths):
            self.index = 0

        return lora_name, full_filename


    class BatchLoraLoader:
        def __init__(self, directory_path, pattern, index):
            self.lora_paths = []
            self.load_loras(directory_path, pattern)
            self.lora_paths.sort()

            self.index = index

        def load_loras(self, directory_path, pattern):
            for file_name in glob.glob(os.path.join(directory_path, pattern), recursive=True):
                temp_strings = file_name.split('\\')
                file_name = temp_strings[len(temp_strings) - 1]

                if file_name.lower().endswith("safetensors"):
                    self.lora_paths.append(file_name)

        def get_lora_by_id(self, lora_id):
            if lora_id < 0 or lora_id >= len(self.lora_paths):
                cstr(f"WarpedLoadLorasBatchByPrefix: Invalid lora index `{lora_id}`").error.print()
                return

            return self.lora_paths[lora_id]

        def get_next_lora(self):
            if self.index >= len(self.lora_paths):
                self.index = 0

            lora_path = self.lora_paths[self.index]
            self.index += 1

            if self.index == len(self.lora_paths):
                self.index = 0

            cstr(f'{cstr.color.YELLOW}{cstr.color.END} Index: {self.index}').msg.print()

            return lora_path

        def get_current_lora(self):
            if self.index >= len(self.lora_paths):
                self.index = 0
            lora_path = self.lora_paths[self.index]

            return lora_path

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

class WarpedHunyuanVideoLoraLoader:
    def __init__(self):
        self.blocks_type = ["all", "single_blocks", "double_blocks"]
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": ("STRING", {"forceInput": True}),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "blocks_type": (["all", "single_blocks", "double_blocks"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "Warped/LORA/Hunyuan"
    OUTPUT_NODE = False
    DESCRIPTION = "LoRA, single blocks double blocks"

    def convert_key_format(self, key: str) -> str:
        prefixes = ["diffusion_model.", "transformer."]
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break

        return key

    def filter_lora_keys(self, lora: Dict[str, torch.Tensor], blocks_type: str) -> Dict[str, torch.Tensor]:
        if blocks_type == "all":
            return lora

        filtered_lora = {}
        for key, value in lora.items():
            base_key = self.convert_key_format(key)

            if blocks_type == "single_blocks" in base_key:
                filtered_lora[key] = value
            elif blocks_type == "double_blocks" in base_key:
                filtered_lora[key] = value

        return filtered_lora

    def check_for_musubi(self, lora: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Checks for and converts from Musubi Tuner format which supports Network Alpha and uses different naming. Largely copied from that project"""
        prefix = "lora_unet_"
        musubi = False
        lora_alphas = {}
        for key, value in lora.items():
            if key.startswith(prefix):
                lora_name = key.split(".", 1)[0]  # before first dot
                if lora_name not in lora_alphas and "alpha" in key:
                    lora_alphas[lora_name] = value
                    musubi = True
        if musubi:
            print("Loading Musubi Tuner format LoRA...")
            converted_lora = {}
            for key, weight in lora.items():
                if key.startswith(prefix):
                    if "alpha" in key:
                        continue
                lora_name = key.split(".", 1)[0]  # before first dot
                # HunyuanVideo lora name to module name: ugly but works
                module_name = lora_name[len(prefix) :]  # remove "lora_unet_"
                module_name = module_name.replace("_", ".")  # replace "_" with "."
                module_name = module_name.replace("double.blocks.", "double_blocks.")  # fix double blocks
                module_name = module_name.replace("single.blocks.", "single_blocks.")  # fix single blocks
                module_name = module_name.replace("img.", "img_")  # fix img
                module_name = module_name.replace("txt.", "txt_")  # fix txt
                module_name = module_name.replace("attn.", "attn_")  # fix attn
                diffusers_prefix = "diffusion_model"
                if "lora_down" in key:
                    new_key = f"{diffusers_prefix}.{module_name}.lora_A.weight"
                    dim = weight.shape[0]
                elif "lora_up" in key:
                    new_key = f"{diffusers_prefix}.{module_name}.lora_B.weight"
                    dim = weight.shape[1]
                else:
                    print(f"unexpected key: {key} in Musubi LoRA format")
                    continue
                # scale weight by alpha
                if lora_name in lora_alphas:
                    # we scale both down and up, so scale is sqrt
                    scale = lora_alphas[lora_name] / dim
                    scale = scale.sqrt()
                    weight = weight * scale
                else:
                    print(f"missing alpha for {lora_name}")

                converted_lora[new_key] = weight
            return converted_lora
        else:
            print("Loading Diffusers format LoRA...")
            return lora

    def load_lora(self, model, lora_name: str, strength: float, blocks_type: str):
        """
        Parameters
        ----------
        model : ModelPatcher
        lora_name : str
        strength : float
        blocks_type : str
            blocks: "all", "single_blocks" "double_blocks"

        Returns
        -------
        tuple
            LoRA
        """
        if not lora_name:
            return (model,)

        from comfy.utils import load_torch_file
        from comfy.sd import load_lora_for_models
        from comfy.lora import load_lora

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise Exception(f"Lora {lora_name} not found at {lora_path}")

        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if self.loaded_lora is None:
            lora = load_torch_file(lora_path)
            self.loaded_lora = (lora_path, lora)

        diffusers_lora = self.check_for_musubi(lora)
        filtered_lora = self.filter_lora_keys(diffusers_lora, blocks_type)

        new_model, _ = load_lora_for_models(model, None, filtered_lora, strength, 0)
        if new_model is not None:
            return (new_model,)

        return (model,)

    @classmethod
    def IS_CHANGED(s, model, lora_name, strength, blocks_type):
        return f"{lora_name}_{strength}_{blocks_type}"

class WarpedFramepackMultiLoraSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora_01": (['None'] + folder_paths.get_filename_list("loras"), {"tooltip": "LORA models are expected to have .safetensors extension"}),
                "strength_01": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "fuse_lora_01": ("BOOLEAN", {"default": False, "tooltip": "Fuse the LORA model with the base model. This is recommended for better performance."}),
               "lora_02": (['None'] + folder_paths.get_filename_list("loras"), {"tooltip": "LORA models are expected to have .safetensors extension"}),
                "strength_02": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "fuse_lora_02": ("BOOLEAN", {"default": False, "tooltip": "Fuse the LORA model with the base model. This is recommended for better performance."}),
               "lora_03": (['None'] + folder_paths.get_filename_list("loras"), {"tooltip": "LORA models are expected to have .safetensors extension"}),
                "strength_03": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "fuse_lora_03": ("BOOLEAN", {"default": False, "tooltip": "Fuse the LORA model with the base model. This is recommended for better performance."}),
               "lora_04": (['None'] + folder_paths.get_filename_list("loras"), {"tooltip": "LORA models are expected to have .safetensors extension"}),
                "strength_04": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "fuse_lora_04": ("BOOLEAN", {"default": False, "tooltip": "Fuse the LORA model with the base model. This is recommended for better performance."}),
            },
            "optional": {
                "prev_lora":("FPLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
            }
        }

    RETURN_TYPES = ("FPLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "select_multiple_loras"
    CATEGORY = "Warped/Framepack/Lora"
    DESCRIPTION = "Select a Hunyuan LoRA model"

    def select_multiple_loras(self, **kwargs):
        loras_list = []

        prev_lora = kwargs.get(f"prev_lora")

        if prev_lora is not None:
            loras_list.extend(prev_lora)

        for i in range(1, 5):
            temp_lora_name = kwargs.get(f"lora_0{i}")
            temp_strength = kwargs.get(f"strength_0{i}")
            temp_fuse_lora = kwargs.get(f"fuse_lora_0{i}")

            # print("lora_name: {}".format(temp_lora_name))
            # print("strength: {}".format(temp_strength))
            # print("fuse_lora: {}".format(temp_fuse_lora))

            if (temp_lora_name != "None") and (Decimal(temp_strength).compare(Decimal(0.0)) != 0):
                lora = {
                    "path": folder_paths.get_full_path("loras", temp_lora_name),
                    "strength": temp_strength,
                    "name": temp_lora_name.split(".")[0],
                    "fuse_lora": temp_fuse_lora,
                }

                loras_list.append(lora)

        if len(loras_list) > 0:
            return (loras_list,)
        else:
            return (None,)

class WarpedHunyuanMultiLoraLoader:
    """
    Hunyuan Multi-Lora Loader
    This node works like the original lora_loader.py, with a required model input and output.
    It does not output LoRA information in HYVIDLORA format.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_01": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_01": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "blocks_type_01": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_02": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_02": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "blocks_type_02": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_03": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_03": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "blocks_type_03": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_04": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_04": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "blocks_type_04": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
            },
          "optional": {
                "lora_name": ("STRING", {"default": None, "forceInput": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "blocks_type": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING", )
    RETURN_NAMES = ("model", "lora_metadata")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "load_multiple_loras"
    CATEGORY = "Warped/LORA/Hunyuan"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

    def convert_key_format(self, key: str) -> str:
        """Standardize LoRA key format by removing prefixes."""
        prefixes = ["diffusion_model.", "transformer."]
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        return key

    def filter_lora_keys(self, lora: Dict[str, torch.Tensor], blocks_type: str) -> Dict[str, torch.Tensor]:
        """Filter LoRA weights based on block type."""
        if blocks_type == "all":
            return lora
        filtered_lora = {}
        for key, value in lora.items():
            base_key = self.convert_key_format(key)
            if blocks_type in base_key:
                filtered_lora[key] = value
        return filtered_lora

    def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load and filter a single LoRA model."""
        if not lora_name or strength == 0:
            return {}, {}

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        # Load the LoRA weights
        lora_weights = utils.load_torch_file(lora_path)

        # Filter the LoRA weights based on the block type
        filtered_lora = self.filter_lora_keys(lora_weights, blocks_type)

        return lora_weights, filtered_lora

#    def load_multiple_loras(self, model, lora_name=None, strength=1.00, blocks_type="all", **kwargs):
    def load_multiple_loras(self, model, **kwargs):
        """Load and apply multiple LoRA models."""
        from comfy.sd import load_lora_for_models

        temp_lora_name = kwargs.get(f"lora_name")
        temp_strength = kwargs.get(f"strength")
        temp_blocks_type = kwargs.get(f"blocks_type")

        lora_metadata = []

        if not temp_lora_name is None and temp_strength != 0:
            print("Lora Name: {}  |  Strength: {}  |  Block Types: {}".format(temp_lora_name, temp_strength, temp_blocks_type))
            lora_metadata.append("{}".format("Lora: {} | Strength: {} | Block Types: {}".format(temp_lora_name, temp_strength, temp_blocks_type)))

            lora_weights, filtered_lora = self.load_lora(temp_lora_name, temp_strength, temp_blocks_type)

            # Apply the LoRA weights to the model
            if filtered_lora:
                model, _ = load_lora_for_models(model, None, filtered_lora, temp_strength, 0)

        for i in range(1, 5):
            temp_lora_name = kwargs.get(f"lora_0{i}")
            temp_strength = kwargs.get(f"strength_0{i}")
            temp_blocks_type = kwargs.get(f"blocks_type_0{i}")

            if temp_lora_name != "None" and temp_strength != 0:
                lora_metadata.append("{}".format("Lora: {} | Strength: {} | Block Types: {}".format(temp_lora_name, temp_strength, temp_blocks_type)))
                # Load and filter the LoRA weights
                lora_weights, filtered_lora = self.load_lora(temp_lora_name, temp_strength, temp_blocks_type)

                # Apply the LoRA weights to the model
                if filtered_lora:
                    model, _ = load_lora_for_models(model, None, filtered_lora, temp_strength, 0)

        return (model, lora_metadata, )

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return f"{kwargs.get('lora_name')}_{kwargs.get('strength')}_{kwargs.get('blocks_type')}_" \
               f"{kwargs.get('lora_01')}_{kwargs.get('strength_01')}_{kwargs.get('blocks_type_01')}_" \
               f"{kwargs.get('lora_02')}_{kwargs.get('strength_02')}_{kwargs.get('blocks_type_02')}_" \
               f"{kwargs.get('lora_03')}_{kwargs.get('strength_03')}_{kwargs.get('blocks_type_03')}_" \
               f"{kwargs.get('lora_04')}_{kwargs.get('strength_04')}_{kwargs.get('blocks_type_04')}"

def get_save_lora_path(filename_prefix, output_dir):
    def map_filename(filename):
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[:prefix_len + 1]

        try:
            temp_strings = filename.split("_")
            temp_strings2 = temp_strings[len(temp_strings) - 1].split('.')
            digits = int(temp_strings2[0])
        except:
            digits = 0

        return (digits, prefix)

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))

    full_output_folder = os.path.join(output_dir, subfolder)

    full_folder_contents = os.listdir(full_output_folder)
    relevant_folder_contents = []

    for temp in full_folder_contents:
        if temp.startswith(filename_prefix):
            relevant_folder_contents.append(temp)

    if len(relevant_folder_contents) > 0:
        try:
            counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_", map(map_filename, relevant_folder_contents)))[0] + 1
        except ValueError:
            counter = 1
        except FileNotFoundError:
            os.makedirs(full_output_folder, exist_ok=True)
            counter = 1
    else:
        counter = 1

    return full_output_folder, filename, counter, subfolder, filename_prefix

class WarpedHunyuanLoraBatchMerge:
    def __init__(self):
        self.base_output_dir = get_default_output_folder()
        os.makedirs(self.base_output_dir, exist_ok = True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_folder": ("STRING", {"default": get_default_output_folder()}),
                "model_prefix": ("STRING", {"default": "new_model_hy"}),
                "lora_1": ("STRING", {"default": None, "forceInput": True}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "blocks_type_1": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_2": (['None'] + get_lora_list(),),
                "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "blocks_type_2": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "merge_multiple_loras"
    CATEGORY = "Warped/HunyuanTools"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

    def convert_key_format(self, key: str) -> str:
        """Standardize LoRA key format by removing prefixes."""
        prefixes = ["diffusion_model.", "transformer."]
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        return key

    def filter_lora_keys(self, lora: Dict[str, torch.Tensor], blocks_type: str) -> Dict[str, torch.Tensor]:
        """Filter LoRA weights based on block type."""
        if blocks_type == "all":
            return lora
        filtered_lora = {}
        for key, value in lora.items():
            base_key = self.convert_key_format(key)
            if blocks_type in base_key:
                filtered_lora[key] = value
        return filtered_lora

    def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load and filter a single LoRA model."""
        if not lora_name or strength == 0:
            return {}, {}

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        # Load the LoRA weights
        lora_weights = utils.load_torch_file(lora_path)

        # Filter the LoRA weights based on the block type
        filtered_lora = self.filter_lora_keys(lora_weights, blocks_type)

        return lora_weights, filtered_lora

    def merge_multiple_loras(self, save_folder, model_prefix, lora_1, strength_1, blocks_type_1, lora_2, strength_2, blocks_type_2, save_metadata=True):
        """Load and apply multiple LoRA models."""
        temp_loras = {}
        metadata = {"loras": "{} and {}".format(lora_1, lora_2)}
        metadata["strengths"] = "{} and {}".format(strength_1, strength_2)
        metadata["block_types"] = "{} and {}".format(blocks_type_1, blocks_type_2)

        print("Processing Lora: {}".format(lora_1))

        if lora_1 != "None" and strength_1 != 0:
            # Load and filter the LoRA weights
            lora_weights, filtered_lora = self.load_lora(lora_1, 1.0, blocks_type_1)
            temp_loras["1"] = {"lora_weights": lora_weights, "strength": strength_1, "filtered_lora": filtered_lora}

        if lora_2 != "None" and strength_2 != 0:
            # Load and filter the LoRA weights
            lora_weights, filtered_lora = self.load_lora(lora_2, 1.0, blocks_type_2)
            temp_loras["2"] = {"lora_weights": lora_weights, "strength": strength_2, "filtered_lora": filtered_lora}

        loras = {}

        for lora_key in temp_loras.keys():
            loras[lora_key] = {"lora_weights": {}, "strength": temp_loras[lora_key]["strength"], "filtered_lora": temp_loras[lora_key]["filtered_lora"]}

            for key in temp_loras[lora_key]["lora_weights"].keys():
                new_key = key.replace("transformer.", "diffusion_model.")
                loras[lora_key]["lora_weights"][new_key] = temp_loras[lora_key]["lora_weights"][key]

        new_lora = {}

        for lora_key in loras.keys():
            for key in loras[lora_key]["lora_weights"].keys():
                if not key in new_lora.keys():
                    new_lora[key] = None
                # print("Lora: {}  | Key: {}  |  Shape: {}".format(lora_key, key, loras[lora_key]["lora_weights"][key].shape))

        # Merge The Weighted Key Weights
        for key in new_lora.keys():
            for lora_key in loras.keys():
                if key in loras[lora_key]["lora_weights"].keys():
                    if not new_lora[key] is None:
                        temp_weights = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"])

                        if new_lora[key].shape[0] < new_lora[key].shape[1]:
                            if temp_weights.shape[0] < new_lora[key].shape[0]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([new_lora[key].shape[0], new_lora[key].shape[1]])
                                padding[:temp_weights.shape[0],:] = temp_weights
                                temp_weights = padding
                            elif temp_weights.shape[0] > new_lora[key].shape[0]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([temp_weights.shape[0], temp_weights.shape[1]])
                                padding[:new_lora[key].shape[0],:] = new_lora[key]
                                new_lora[key] = padding
                        else:
                            if temp_weights.shape[1] < new_lora[key].shape[1]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([new_lora[key].shape[0], new_lora[key].shape[1]])
                                padding[:,:temp_weights.shape[1]] = temp_weights
                                temp_weights = padding
                            elif temp_weights.shape[1] > new_lora[key].shape[1]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([temp_weights.shape[0], temp_weights.shape[1]])
                                padding[:,:new_lora[key].shape[1]] = new_lora[key]
                                new_lora[key] = padding

                        try:
                            new_lora[key] = torch.add(new_lora[key], temp_weights)
                        except Exception as e:
                            raise(e)
                    else:
                        new_lora[key] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"])

        if not save_metadata:
            metadata = None

        full_output_path, filename, counter, subfolder, filename_prefix = get_save_lora_path(model_prefix, self.base_output_dir)

        output_filename = os.path.join(full_output_path, "{}_{:05}.safetensors".format(model_prefix, counter))
        utils.save_torch_file(new_lora, output_filename, metadata=metadata)

        save_message = "Weights Saved To: {}".format(output_filename)
        print(save_message)

        return {"ui": {"tags": [save_message]}}

class WarpedHunyuanLoraConvertKeys:
    def __init__(self):
        self.base_output_dir = get_default_output_folder()
        os.makedirs(self.base_output_dir, exist_ok = True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_path": ("STRING", {"default": get_default_output_path()}),
                "lora": (get_lora_list(),),
                "convert_to": (["diffusion_model", "transformer"], {"default": "diffusion_model"}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert_lora"
    CATEGORY = "Warped/HunyuanTools"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

    def check_for_musubi(self, lora: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Checks for and converts from Musubi Tuner format which supports Network Alpha and uses different naming. Largely copied from that project"""
        prefix = "lora_unet_"
        musubi = False
        lora_alphas = {}
        for key, value in lora.items():
            if key.startswith(prefix):
                lora_name = key.split(".", 1)[0]  # before first dot
                if lora_name not in lora_alphas and "alpha" in key:
                    lora_alphas[lora_name] = value
                    musubi = True
        if musubi:
            print("Loading Musubi Tuner format LoRA...")
            converted_lora = {}
            for key, weight in lora.items():
                if key.startswith(prefix):
                    if "alpha" in key:
                        continue
                lora_name = key.split(".", 1)[0]  # before first dot
                # HunyuanVideo lora name to module name: ugly but works
                module_name = lora_name[len(prefix) :]  # remove "lora_unet_"
                module_name = module_name.replace("_", ".")  # replace "_" with "."
                module_name = module_name.replace("double.blocks.", "double_blocks.")  # fix double blocks
                module_name = module_name.replace("single.blocks.", "single_blocks.")  # fix single blocks
                module_name = module_name.replace("img.", "img_")  # fix img
                module_name = module_name.replace("txt.", "txt_")  # fix txt
                module_name = module_name.replace("attn.", "attn_")  # fix attn
                diffusers_prefix = "diffusion_model"
                if "lora_down" in key:
                    new_key = f"{diffusers_prefix}.{module_name}.lora_A.weight"
                    dim = weight.shape[0]
                elif "lora_up" in key:
                    new_key = f"{diffusers_prefix}.{module_name}.lora_B.weight"
                    dim = weight.shape[1]
                else:
                    print(f"unexpected key: {key} in Musubi LoRA format")
                    continue
                # scale weight by alpha
                if lora_name in lora_alphas:
                    # we scale both down and up, so scale is sqrt
                    scale = lora_alphas[lora_name] / dim
                    scale = scale.sqrt()
                    weight = weight * scale
                else:
                    print(f"missing alpha for {lora_name}")

                converted_lora[new_key] = weight
            return converted_lora
        else:
            print("Loading Diffusers format LoRA...")
            return lora

    def load_lora(self, lora_name: str) -> Tuple[Dict[str, torch.Tensor],]:
        """Load and filter a single LoRA model."""
        if not lora_name:
            return {}

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        # Load the LoRA weights
        lora_weights = utils.load_torch_file(lora_path)

        return lora_weights

    def convert_lora(self, save_path, lora, convert_to, save_metadata=True):
        metadata = {"original_lora": "{}".format(lora)}

        # Load the LoRA weights
        temp_lora = self.load_lora(lora)
        temp_lora = self.check_for_musubi(temp_lora)

        new_lora = {}

        for key in temp_lora.keys():
            if key.startswith("transformer.") and  (convert_to == "diffusion_model"):
                new_key = key.replace("transformer.", "diffusion_model.")
                new_lora[new_key] = temp_lora[key]
                continue

            if key.startswith("diffusion_model.") and  (convert_to == "transformer"):
                new_key = key.replace("diffusion_model.", "transformer.")
                new_lora[new_key] = temp_lora[key]
                continue

            if key.startswith("lora_unet_"):
                new_key = key.replace("lora_unet_", "{}.".format(convert_to))

                if "double" in new_key:
                    new_key = new_key.replace("double_blocks_", "double_blocks.")
                    new_key = new_key.replace("_img_attn", ".img_attn")
                    new_key = new_key.replace("_img_mlp", ".img_mlp")
                    new_key = new_key.replace("_txt_attn", ".txt_attn")
                    new_key = new_key.replace("_txt_mlp", ".txt_mlp")
                    new_key = new_key.replace(".lora_up.", ".lora_A.")
                    new_key = new_key.replace(".lora_down.", ".lora_B.")

                    continue

                if "single" in new_key:
                    new_key = new_key.replace("single_blocks_", "single_blocks.")
                    new_key = new_key.replace("_linear", ".linear")
                    new_key = new_key.replace(".lora_up.", ".lora_A.")
                    new_key = new_key.replace(".lora_down.", ".lora_B.")

                    continue

        if not save_metadata:
            metadata = None

        if len(new_lora) < 1:
            utils.save_torch_file(temp_lora, save_path, metadata=metadata)
        else:
            utils.save_torch_file(new_lora, save_path, metadata=metadata)

        save_message = "Weights Saved To: {}".format(save_path)

        return {"ui": {"tags": [save_message]}}

class WarpedLoraKeysAndMetadataReader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", )
    RETURN_NAMES = ("keys", "metadata", )
    OUTPUT_IS_LIST = (True, True, )
    FUNCTION = "read_data"
    CATEGORY = "Warped/Utils"
    DESCRIPTION = "Read Metadata From Lora."

    def get_metadata(self, lora_path):
        # Open the file in binary mode
        with open(lora_path, 'rb') as file:
            length_of_header_bytes = file.read(8)
            # Interpret the bytes as a little-endian unsigned 64-bit integer
            length_of_header = struct.unpack('<Q', length_of_header_bytes)[0]
            header_bytes = file.read(length_of_header)
            #header = json.loads(header_bytes.decode('utf-8'))
            metadata = json.loads(header_bytes)
            return metadata["__metadata__"]

    def load_lora(self, lora_name: str, strength: 1.0, blocks_type: "all") -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load and filter a single LoRA model."""
        if not lora_name:
            return "", ""

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        lora_weights = comfy.utils.load_torch_file(lora_path)

        metadata = self.get_metadata(lora_path)

        return lora_weights, metadata

    def read_data(self, lora_name):
        metadata = {}
        keys = []

        if lora_name != "None":
            # Load and filter the LoRA weights
            lora_weights, metadata = self.load_lora(lora_name, 1.0, "all")

        if metadata is None:
            metadata = {}

        lora_keys = []
        for key in lora_weights.keys():
            lora_keys.append("{}".format(key))
            # print(key)

        lora_metadata = []
        if len(metadata.keys()) > 0:
            for key in metadata.keys():
                lora_metadata.append("{}: {}".format(key, metadata[key]))
                # print("{}: {}".format(key, metadata[key]))

        lora_weights = None

        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        return { "ui": { "lora_keys": lora_keys, "lora_metadata": lora_metadata }, "result": (lora_keys, lora_metadata,), }

class WarpedHunyuanLoraConvert:
    def __init__(self):
        self.base_output_dir = get_default_output_folder()
        os.makedirs(self.base_output_dir, exist_ok = True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_path": ("STRING", {"default": get_default_output_path()}),
                "lora": (get_lora_list(),),
                "convert_to": (['32', '64', '128'], {"default": "32"}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert_lora"
    CATEGORY = "Warped/HunyuanTools"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

    def load_lora(self, lora_name: str) -> Tuple[Dict[str, torch.Tensor]]:
        if not lora_name:
            return {}

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        # Load the LoRA weights
        lora_weights = utils.load_torch_file(lora_path)

        return lora_weights

    def convert_lora(self, save_path, lora, convert_to="32", save_metadata=True):
        metadata = {"original_lora": "{}".format(lora)}

        if lora != "None":
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora)

        for key in lora_weights.keys():
            sample_shape = lora_weights[key].shape
            print("Lora Shape Before: {}".format(sample_shape))

            if sample_shape[0] < sample_shape[1]:
                sample_res = int(sample_shape[0])
            else:
                sample_res = int(sample_shape[1])

            break

        target_res = int(convert_to)

        if target_res == sample_res:
            save_message = "LORA Resolution and Target Resolution are the same. Nothing to convert."
            print(save_message)
            return {"ui": {"tags": [save_message]}}

        new_lora = {}

        # Convert The LORA Weights
        for key in lora_weights.keys():
            temp_weights = lora_weights[key].clone().detach()

            if temp_weights.shape[0] < temp_weights.shape[1]:
                padding = torch.zeros([target_res, temp_weights.shape[1]])

                # if upscale
                if temp_weights.shape[0] < target_res:
                    padding[:temp_weights.shape[0],:] = temp_weights
                    new_lora[key] = padding
                # if downscale
                else:
                    padding[:target_res,:] = temp_weights[:target_res,:]
                    new_lora[key] = padding
            else:
                padding = torch.zeros([temp_weights.shape[0], target_res])

                # if upscale
                if temp_weights.shape[1] < target_res:
                    padding[:,:temp_weights.shape[1]] = temp_weights
                    new_lora[key] = padding
                # if downscale
                else:
                    padding[:,:target_res] = temp_weights[:,:target_res]
                    new_lora[key] = padding

        if not save_metadata:
            metadata = None

        utils.save_torch_file(new_lora, save_path, metadata=metadata)

        save_message = "Weights Saved To: {}".format(save_path)

        return {"ui": {"tags": [save_message]}}

vae_scaling_factor = 0.476986

class WarpedFramepackSampler:
    def __init__(self):
        self.clip_vision = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "steps": ("INT", {"default": 25, "min": 1}),
                "use_teacache": ("BOOLEAN", {"default": True, "tooltip": "Use teacache for faster sampling."}),
                "teacache_rel_l1_thresh": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The threshold for the relative L1 loss."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 32.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "preferred_batch_size": ("INT", {"default": 61, "min": 37, "max": 161, "step": 4, "tooltip": "The preferred number of frames to use for sampling."}),
                "use_batch_size": (["next_lowest", "next_highest", "closest", "exact"], {"default": "next_lowest", "tooltip": "Number of frames generated may be impacted by choice."}),
                "total_second_length": ("FLOAT", {"default": 10, "min": 1, "max": 3600, "step": 0.1, "tooltip": "For I2V and T2V, The total length of the video in seconds. Disreguarded for V2V"}),
                "gpu_memory_preservation": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 128.0, "step": 0.1, "tooltip": "The amount of GPU memory to preserve."}),
                "sampler": (["unipc_bh1", "unipc_bh2"],
                    {
                        "default": 'unipc_bh1'
                    }),
                "dec_tile_size": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 32}),
                "dec_overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "dec_temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to decode at a time."}),
                "dec_temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap."}),
                "skip_frames": ("INT", {"default": 0, "min": 0, "max": 128, "step": 4, "tooltip": "Number of frames to skip from beginning for output."}),
                "clip_vision_model": ("CLIP_VISION", ),
            },
            "optional": {
                "start_image": ("IMAGE", {"tooltip": "init image to use for image2video or video2video"} ),
                "end_image": ("IMAGE", {"tooltip": "end image to use for image2video"} ),
                "embed_interpolation": (["disabled", "weighted_average", "linear"], {"default": 'disabled', "tooltip": "Image embedding interpolation type. If linear, will smoothly interpolate with time, else it'll be weighted average with the specified weight."}),
                "start_embed_strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Weighted average constant for image embed interpolation. If end image is not set, the embed's strength won't be affected"}),
                "video_image_batch": ("IMAGE", {"tooltip": "init Latents to use for video2video"} ),
                "fps": ("FLOAT", {"default": 30.00}),
                "denoise_strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "noise_strength": ("FLOAT", {"default": 1.00, "min": 0.10, "max": 1.00, "step": 0.01}),
                "blend_frames": ("INT", {"default":1, "min":0, "max": 3, "step": 1}),
                "t2v_width": ("INT", {"default":640, "min":256, "max": 1280, "step": 8}),
                "t2v_height": ("INT", {"default":640, "min":256, "max": 1280, "step": 8}),
                "v2v_context_count": ("INT", {"default":5, "min":3, "max": 10, "step": 1}),
                "verbose_messaging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", )
    RETURN_NAMES = ("images", "generation_status", "fps", )
    FUNCTION = "process"
    CATEGORY = "Warped/Framepack/Sampling"

    def process(self, model, vae, shift, positive, negative, preferred_batch_size, use_batch_size, use_teacache, total_second_length, teacache_rel_l1_thresh, steps, cfg,
                guidance_scale, seed, sampler, dec_tile_size, dec_overlap, dec_temporal_size, dec_temporal_overlap, skip_frames, clip_vision_model,
                gpu_memory_preservation, start_image=None, end_image=None, embed_interpolation="linear", start_embed_strength=1.0, video_image_batch=None,
                fps=30.00, denoise_strength=1.00, noise_strength=1.00, blend_frames=0, t2v_width=640, t2v_height=640, v2v_context_count=5, verbose_messaging=False):
        self.dec_tile_size = dec_tile_size
        self.dec_overlap = dec_overlap
        self.dec_temporal_size = dec_temporal_size
        self.dec_temporal_overlap = dec_temporal_overlap
        self.skip_frames = skip_frames
        self.vae = vae
        self.total_second_length = total_second_length
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.use_teacache = use_teacache
        self.teacache_rel_l1_thresh = teacache_rel_l1_thresh
        self.guidance_scale = guidance_scale
        self.sampler = sampler
        self.skip_frames = skip_frames
        self.gpu_memory_preservation = gpu_memory_preservation
        self.embed_interpolation = embed_interpolation
        self.start_embed_strength = start_embed_strength
        self.denoise_strength = denoise_strength
        self.noise_strength = noise_strength
        self.transformer = model["transformer"]
        self.base_dtype = model["dtype"]
        self.positive = positive
        self.negative = negative
        self.device = mm.get_torch_device()
        self.offload_device = mm.unet_offload_device()
        self.clip_vision = clip_vision_model
        self.blend_frames = blend_frames
        self.shift = shift
        self.t2v_width=t2v_width
        self.t2v_height=t2v_height
        self.fps = fps
        self.v2v_context_count = v2v_context_count
        self.verbose_messaging = verbose_messaging

        if not video_image_batch is None:
            if len(video_image_batch.shape) < 4:
                video_image_batch = video_image_batch.unsqueeze(0)

            self.latent_window_size, self.batch_count, truncated_frame_count = self.get_latent_window_size(preferred_batch_size, video_image_batch.shape[0], use_batch_size=use_batch_size)
            self.total_second_length = truncated_frame_count / self.fps

            if video_image_batch.shape[0] != truncated_frame_count:
                image_batches_tuple = torch.split(video_image_batch, truncated_frame_count, dim=0)
                image_batches_split = [item for item in image_batches_tuple]
                video_image_batch = image_batches_split[0]
        else:
            frame_count = int(((total_second_length * self.fps) / 4) * 4) + 1
            print("frame_count: {}".format(frame_count))
            self.latent_window_size, self.batch_count, truncated_frame_count = self.get_latent_window_size(preferred_batch_size, frame_count, use_batch_size=use_batch_size)
            self.total_second_length = truncated_frame_count / self.fps

        self.total_frames = truncated_frame_count

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        if not video_image_batch is None:
            return self.process_v2v(start_image, video_image_batch)

        if not start_image is None:
            return self.process_i2v(start_image, end_image, is_i2v=True)

        return self.process_t2v()

    def process_t2v(self):
        return self.process_i2v(None, None, is_i2v=False)

    def process_i2v(self, start_image, end_image, is_i2v=True):
        start_latent = None
        end_latent = None

        if is_i2v:
            if not start_image is None:
                image_embeds = self.clip_vision_encode(start_image)
                start_latent = self.encode_batched(start_image, per_batch=self.latent_window_size)

                start_latent = start_latent["samples"] * vae_scaling_factor
                print("start_latent Shape: {}".format(start_latent.shape))

            if not end_image is None:
                if (start_image.shape[1] != end_image.shape[1]) or (start_image.shape[2] != end_image.shape[2]):
                    raise ValueError("Unable to continue: end_image Height/Width Does Not Match start_image Height/Width.")

                end_latent = self.encode_batched(end_image, per_batch=self.latent_window_size)
                print("end_latent Shape: {}".format(end_latent["samples"].shape))
                end_image_embeds = self.clip_vision_encode(end_image)
        else:
            start_image = torch.zeros(size=(1, self.t2v_height, self.t2v_width, 3), dtype=torch.float32).cpu()
            start_latent = torch.zeros(size=(1, 16, 1, int(self.t2v_height // 8), int(self.t2v_width // 8)), dtype=torch.float32).cpu()
            image_embeds = self.clip_vision_encode(start_image)

        self.width = start_image.shape[2]
        self.height = start_image.shape[1]

        total_latent_sections = self.batch_count

        if self.verbose_messaging:
            print("total_latent_sections: ", total_latent_sections)

        if end_latent is not None:
            end_latent = end_latent["samples"] * vae_scaling_factor
        has_end_image = end_latent is not None

        if self.verbose_messaging:
            print("start_latent", start_latent.shape)

        B, C, T, H, W = start_latent.shape

        if self.verbose_messaging:
            print("process_i2v: Image Width: {}".format(self.width))
            print("process_i2v: Image Width: {}".format(self.width))
            print("process_i2v: total_frames: {}".format(self.total_frames))
            print("process_i2v: total_second_length: {}".format(self.total_second_length))
            print("process_i2v: batch_count: {}".format(self.batch_count))
            print("process_i2v: latent_window_size: {}".format(self.latent_window_size))

        start_image_encoder_last_hidden_state = image_embeds["last_hidden_state"].to(self.base_dtype).to(self.device)

        if has_end_image:
            assert end_image_embeds is not None
            end_image_encoder_last_hidden_state = end_image_embeds["last_hidden_state"].to(self.base_dtype).to(self.device)
        else:
            end_image_encoder_last_hidden_state = torch.zeros_like(start_image_encoder_last_hidden_state)

        llama_vec = self.positive[0][0].to(self.base_dtype).to(self.device)
        clip_l_pooler = self.positive[0][1]["pooled_output"].to(self.base_dtype).to(self.device)

        if not math.isclose(self.cfg, 1.0):
            llama_vec_n = self.negative[0][0].to(self.base_dtype)
            clip_l_pooler_n = self.negative[0][1]["pooled_output"].to(self.base_dtype).to(self.device)
        else:
            llama_vec_n = torch.zeros_like(llama_vec, device=self.device)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler, device=self.device)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        rnd = torch.Generator("cpu").manual_seed(self.seed)

        num_frames = self.encoded_to_decoded_length(self.latent_window_size)

        if self.verbose_messaging:
            print("num_frames: {}".format(num_frames))

        total_generated_latent_frames = 0

        if is_i2v:
            if start_latent is None:
                raise ValueError("A start_image value is required for I2V.")

            cat_list = []

            history_latents = torch.zeros(size=(1, 16, 19, H, W), dtype=torch.float32).cpu()

            cat_list.append(start_latent.to(history_latents))
            cat_list.append(history_latents)

            history_latents = torch.cat(cat_list, dim=2)
        else:
            history_latents = torch.zeros(size=(1, 16, 20, H, W), dtype=torch.float32).cpu()

        original_history_latents = history_latents.clone().detach()

        latent_paddings_list = list(range(total_latent_sections))
        latent_paddings = latent_paddings_list.copy()  # Create a copy for iteration

        comfy_model = HyVideoModel(
                HyVideoModelConfig(self.base_dtype),
                model_type=comfy.model_base.ModelType.FLOW,
                device=self.device,
            )

        patcher = comfy.model_patcher.ModelPatcher(comfy_model, self.device, torch.device("cpu"))
        callback = latent_preview.prepare_callback(patcher, self.steps)

        move_model_to_device_with_memory_preservation(self.transformer, target_device=self.device, preserved_memory_gb=self.gpu_memory_preservation)

        if self.verbose_messaging:
            print("latent_paddings: {}".format(latent_paddings))

        latent_batches_gend = []

        temp_noise = torch.randn((int(llama_vec.shape[0]), 16, self.latent_window_size * self.batch_count, H, W), generator=rnd, device=rnd.device).to(device=rnd.device, dtype=torch.float32)

        if Decimal(self.noise_strength).compare(Decimal(1.00)) != 0:
            temp_noise = torch.mul(temp_noise, self.noise_strength)

        temp_noise_tuple = torch.split(temp_noise, self.latent_window_size, dim=2)
        noise = [item for item in temp_noise_tuple]

        temp_noise = None
        temp_noise_tuple = None
        generated_latents = None
        interrupted = False
        real_history_latents = None
        context_latents = None
        generation_status = ""
        temp_history_latents = None

        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        try:
            for i, latent_padding in enumerate(latent_paddings):
                is_first_section = latent_padding == min(latent_paddings)
                is_last_section = latent_padding == max(latent_paddings)

                if self.verbose_messaging:
                    print("history_latents Shape: {}".format(history_latents.shape))

                noise_latent = noise[latent_padding]

                latent_padding_size = latent_padding * self.latent_window_size

                if self.verbose_messaging:
                    print("latent_padding_size {}: {}  |  latent_padding: {} latent_window_size: {}".format(latent_padding, latent_padding_size, latent_padding, self.latent_window_size))

                if self.embed_interpolation != "disabled":
                    if self.embed_interpolation == "linear":
                        if total_latent_sections <= 1:
                            frac = 1.0  # Handle case with only one section
                        else:
                            frac = 1 - (i / (total_latent_sections - 1))  # going backwards
                    else:
                        frac = self.start_embed_strength if has_end_image else 1.0

                    image_encoder_last_hidden_state = ((start_image_encoder_last_hidden_state * frac) + ((1 - frac) * end_image_encoder_last_hidden_state)) * self.start_embed_strength
                else:
                    image_encoder_last_hidden_state = start_image_encoder_last_hidden_state * self.start_embed_strength

                start_latent_frames = T  # 0 or 1

                if self.verbose_messaging:
                    print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

                indices = torch.arange(0, sum([1, 16, 2, 1, self.latent_window_size])).unsqueeze(0)
                clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, self.latent_window_size], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)
                clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)

                if is_last_section and (not end_latent is None):
                    clean_latents = torch.cat([start_latent.to(history_latents), end_latent.to(history_latents)], dim=2)
                else:
                    clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

                if self.use_teacache:
                    self.transformer.initialize_teacache(enable_teacache=True, num_steps=self.steps, rel_l1_thresh=self.teacache_rel_l1_thresh)
                else:
                    self.transformer.initialize_teacache(enable_teacache=False)

                generated_latents = self.generate_video(W * 8, H * 8, num_frames, rnd, noise_latent, llama_vec, llama_attention_mask, clip_l_pooler, llama_vec_n, llama_attention_mask_n, clip_l_pooler_n, image_encoder_last_hidden_state,
                                                        latent_indices, clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices, callback)

                noise[latent_padding] = None
                print("generated_latents {} | Shape: {}".format(latent_padding, generated_latents.shape))

                if not is_last_section:
                    history_latents = torch.cat([original_history_latents, generated_latents.to(original_history_latents)], dim=2)

                latent_batches_gend.append(generated_latents.clone().detach()  / vae_scaling_factor)

                if self.verbose_messaging:
                    print("history_latents {} | Shape: {}".format(latent_padding, history_latents.shape))

                total_generated_latent_frames += int(generated_latents.shape[2])

                if is_last_section:
                    break
        except mm.InterruptProcessingException as ie:
            interrupted = True
            print(f"\nWarpedSamplerCustomAdv: Processing Interrupted.")
            print("WarpedSamplerCustomAdv: Returning only partial results (if any).\n If zero images generated, a blank yellow image will be returned.\n")

            generation_status = f"\nWarpedSamplerCustomAdv: Processing Interrupted."

            pass
        except BaseException as e:
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

            print(f"\nWarpedSamplerCustomAdv: Exception During Processing: {str(e)}")
            print("WarpedSamplerCustomAdv: Returning only partial results (if any).\n If zero images generated, a blank red image will be returned.\n")
            generation_status = f"WarpedSamplerCustomAdv: Exception During Processing: {str(e)}"
            generation_status = "{}{}".format(generation_status, "WarpedSamplerCustomAdv: Returning only partial results (if any).\nIf zero images generated, a blank red image will be returned.")

            traceback.print_tb(e.__traceback__, limit=99, file=sys.stdout)

            pass

        history_latents= None
        latent_paddings = None
        noise = None
        indices = None
        clean_latent_indices_pre = None
        blank_indices = None
        latent_indices = None
        clean_latent_indices_post = None
        clean_latent_2x_indices = None
        clean_latent_4x_indices = None
        clean_latent_indices = None
        clean_latents_pre = None
        clean_latents_post = None
        clean_latents_2x = None
        clean_latents_4x = None
        clean_latents = None
        clean_latent_indices_start = None
        clean_latent_1x_indices = None
        clean_latents_1x = None
        video_latent_batches = None
        video_image_batch = None

        self.transformer.to(self.offload_device)
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        if len(latent_batches_gend) > 0:
            output_images = self.decode_batches(latent_batches_gend, self.skip_frames)
        elif interrupted:
            temp_image = Image.new('RGB', (self.width, self.height), color = 'yellow')
            output_images = pil2tensorSwap(temp_image)
        else:
            temp_image = Image.new('RGB', (self.width, self.height), color = 'red')
            output_images = pil2tensorSwap(temp_image)

        latent_batches_gend = None

        generation_status = "{}\nImages Generated: {}".format(generation_status, output_images.shape[0])

        if self.verbose_messaging:
            print("output_images Shape: {}".format(output_images.shape))

        return (output_images, generation_status, self.fps, )

    def process_v2v(self, start_image, video_image_batch):
        if len(video_image_batch.shape) < 4:
            video_image_batch = video_image_batch.unsqueeze(0)

        self.width = video_image_batch.shape[2]
        self.height = video_image_batch.shape[1]
        self.buffer_length = 19

        image_batch_size = int((video_image_batch.shape[0] - 1) / self.batch_count) + 1
        latent_size_factor = 4
        latent_batch_size = self.decoded_to_encoded_length(image_batch_size)

        original_start_latent = None

        if not start_image is None:
            original_start_latent = self.encode_batched(start_image, self.latent_window_size)
            original_start_latent = original_start_latent["samples"] * vae_scaling_factor
            original_start_latent_embedding = self.clip_vision_encode(start_image)

        rnd = torch.Generator("cpu").manual_seed(self.seed)

        if self.verbose_messaging:
            print("process_v2v: Video Context Images Shape: {}".format(video_image_batch.shape))
            print("process_v2v: image_batch_size: {}".format(image_batch_size))
            print("process_v2v: latent_batch_size: {}".format(latent_batch_size))
            print("process_v2v: Total Seconds Length: {}".format(self.total_second_length))

        total_latent_sections = self.batch_count

        video_image_batch.to(dtype=torch.float32, device=self.device)
        video_latent_batches, video_encoding_batches = self.video_encode(video_image_batch, image_batch_size)
        video_image_batch.to(dtype=torch.float32, device=self.offload_device)

        B, C, _, H, W = video_latent_batches[0].shape
        T = 1

        llama_vec = self.positive[0][0].to(self.base_dtype).to(self.device)
        clip_l_pooler = self.positive[0][1]["pooled_output"].to(self.base_dtype).to(self.device)

        if not math.isclose(self.cfg, 1.0):
            llama_vec_n = self.negative[0][0].to(self.base_dtype)
            clip_l_pooler_n = self.negative[0][1]["pooled_output"].to(self.base_dtype).to(self.device)
        else:
            llama_vec_n = torch.zeros_like(llama_vec, device=self.device)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler, device=self.device)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        output_images = None
        num_frames = self.encoded_to_decoded_length(self.latent_window_size)
        latent_paddings_list = list(range(total_latent_sections))

        total_generated_latent_frames = 0
        latent_paddings = latent_paddings_list.copy()  # Create a copy for iteration

        latent_embeds = self.get_video_latent_embeds(video_image_batch, image_batch_size, latent_paddings)

        comfy_model = HyVideoModel(
                HyVideoModelConfig(self.base_dtype),
                model_type=comfy.model_base.ModelType.FLOW,
                device=self.device,
            )

        patcher = comfy.model_patcher.ModelPatcher(comfy_model, self.device, torch.device("cpu"))
        callback = latent_preview.prepare_callback(patcher, self.steps)

        move_model_to_device_with_memory_preservation(self.transformer, target_device=self.device, preserved_memory_gb=self.gpu_memory_preservation)

        print("latent_paddings count: {}  |  latent_paddings: {}".format(len(latent_paddings), latent_paddings))

        latent_batches_gend = []

        temp_noise = torch.randn((int(llama_vec.shape[0]), 16, self.latent_window_size * self.batch_count, H, W), generator=rnd, device=rnd.device).to(device=rnd.device, dtype=torch.float32)

        if Decimal(self.noise_strength).compare(Decimal(1.00)) != 0:
            temp_noise = torch.mul(temp_noise, self.noise_strength)

        temp_noise_tuple = torch.split(temp_noise, self.latent_window_size, dim=2)
        noise = [item for item in temp_noise_tuple]

        if self.verbose_messaging:
            print("Noise Batches Length: {}".format(len(noise)))
            print("sample noise Shape: {}".format(noise[0].shape))

        temp_noise = None
        temp_noise_tuple = None
        has_end_image = True
        generated_latents = None
        interrupted = False
        history_latents = None
        real_history_latents = None
        context_latents = None
        generation_status = ""
        temp_history_latents = None
        original_history_latents = None

        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        try:
            for padding_i, latent_padding in enumerate(latent_paddings):
                padding_key = "{}".format(latent_padding)
                prev_padding_key = "{}".format(latent_padding - 1)
                next_padding_key = "{}".format(latent_padding + 1)

                is_first_section = latent_padding == min(latent_paddings)
                is_last_section = latent_padding == max(latent_paddings)

                start_latent = latent_embeds[padding_key]["start_latent"]
                start_image_encoder_last_hidden_state = latent_embeds[padding_key]["start_embedding"]["last_hidden_state"].to(self.base_dtype).to(self.device)
                image_encoder_last_hidden_state = torch.mul(start_image_encoder_last_hidden_state, self.start_embed_strength)

                if self.verbose_messaging:
                    print("latent_padding: {}  |  latent_window_size: {}".format(latent_padding, self.latent_window_size))

                noise_latent = noise[latent_padding]

                if self.verbose_messaging:
                    print("noise_latent Shape: ()".format(noise_latent.shape))

                if is_first_section:
                    if original_start_latent is None:
                        original_history_latents = torch.zeros(size=(1, 16, self.buffer_length, H, W), dtype=torch.float32).cpu()
                    else:
                        original_history_latents = torch.zeros(size=(1, 16, self.buffer_length - 1, H, W), dtype=torch.float32).cpu()
                        original_history_latents = torch.cat([original_start_latent.to(original_history_latents), original_history_latents], 2)
                        original_start_embedding_hidden_states = original_start_latent_embedding["last_hidden_state"].to(self.base_dtype).to(self.device)

                if not original_start_latent is None:
                    image_encoder_last_hidden_state = torch.mul(original_start_embedding_hidden_states, self.start_embed_strength)

                latent_padding_size = 0
                start_latent_frames = 1
                print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

                history_latents = torch.cat([original_history_latents, video_latent_batches[latent_padding].to(original_history_latents)], dim=2)

                is_plus_one = False
                indices = torch.arange(0, history_latents.shape[2]).unsqueeze(0)

                clean_latents, clean_latent_indices, latent_indices = self.get_video_clean_latents(history_latents, self.v2v_context_count, is_plus_one=is_plus_one)
                clean_latent_4x_indices, clean_latent_2x_indices, blank_indices = indices.split([history_latents.shape[2] - 2 - video_latent_batches[latent_padding].shape[2], 2, video_latent_batches[latent_padding].shape[2]], dim=1)
                clean_latents_4x, clean_latents_2x, blank_latents = history_latents[:, :, :history_latents.shape[2], :, :].split([history_latents.shape[2] - 2 - video_latent_batches[latent_padding].shape[2], 2, video_latent_batches[latent_padding].shape[2]], dim=2)

                if self.v2v_context_count > history_latents.shape[2]:
                    self.v2v_context_count = history_latents.shape[2]

                num_frames = self.encoded_to_decoded_length(self.latent_window_size)

                if self.verbose_messaging:
                    print("history_latents Shape: {}\n".format(history_latents.shape))

                    print("indices: {}".format(indices))
                    print("latent_indices: {}\n".format(latent_indices))
                    print("clean_latent_2x_indices: {}".format(clean_latent_2x_indices))
                    print("clean_latent_4x_indices: {}".format(clean_latent_4x_indices))
                    print("clean_latent_indices: {}\n".format(clean_latent_indices))

                    print("clean_latents_2x Shape: {}".format(clean_latents_2x.shape))
                    print("clean_latents_4x Shape: {}".format(clean_latents_4x.shape))
                    print("clean_latents Shape: {}\n".format(clean_latents.shape))
                    print("noise Shape: {}\n".format(noise[latent_padding].shape))

                if self.use_teacache:
                    self.transformer.initialize_teacache(enable_teacache=True, num_steps=self.steps, rel_l1_thresh=self.teacache_rel_l1_thresh)
                else:
                    self.transformer.initialize_teacache(enable_teacache=False)

                clean_latents.to(dtype=torch.float32, device=self.device)
                clean_latents_2x.to(dtype=torch.float32, device=self.device)
                clean_latents_4x.to(dtype=torch.float32, device=self.device)
                noise_latent.to(dtype=torch.float32, device=self.device)

                generated_latents = self.generate_video(W * 8, H * 8, num_frames, rnd, noise_latent, llama_vec, llama_attention_mask, clip_l_pooler, llama_vec_n, llama_attention_mask_n, clip_l_pooler_n, image_encoder_last_hidden_state,
                                                        latent_indices, clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices, callback)

                noise[latent_padding] = None
                print("generated_latents for batch {}: Shape: {}\n".format(latent_padding, generated_latents.shape))
                latent_batches_gend.append(generated_latents.clone().detach()  / vae_scaling_factor)

                total_generated_latent_frames += int(generated_latents.shape[2])

                generated_latents.to(dtype=torch.float32, device=self.offload_device)
                generated_latents = None

                clean_latents = clean_latents.to(dtype=torch.float32, device=self.offload_device)
                clean_latents_2x = clean_latents_2x.to(dtype=torch.float32, device=self.offload_device)
                clean_latents_4x = clean_latents_4x.to(dtype=torch.float32, device=self.offload_device)
                noise_latent = noise_latent.to(dtype=torch.float32, device=self.offload_device)

                initial_latent = None
                clean_latents = None
                clean_latents_2x = None
                clean_latents_4x = None
                noise_latent = None
                history_latents = None
                history_latents2 = None

                video_latent_batches[latent_padding] = video_latent_batches[latent_padding].to(device=self.offload_device)
                video_latent_batches[latent_padding] = None

                image_encoder_last_hidden_state = None
                generated_latents = None

                mm.soft_empty_cache()
                gc.collect()
                time.sleep(1)

                if is_last_section:
                    break
        except mm.InterruptProcessingException as ie:
            interrupted = True
            print(f"\nWarpedSamplerCustomAdv: Processing Interrupted.")
            print("WarpedSamplerCustomAdv: Returning only partial results (if any).\n If zero images generated, a blank yellow image will be returned.\n")

            generation_status = f"\nWarpedSamplerCustomAdv: Processing Interrupted."

            pass
        except BaseException as e:
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

            print(f"\nWarpedSamplerCustomAdv: Exception During Processing: {str(e)}")
            print("WarpedSamplerCustomAdv: Returning only partial results (if any).\n If zero images generated, a blank red image will be returned.\n")
            generation_status = f"WarpedSamplerCustomAdv: Exception During Processing: {str(e)}"
            generation_status = "{}{}".format(generation_status, "WarpedSamplerCustomAdv: Returning only partial results (if any).\nIf zero images generated, a blank red image will be returned.")

            traceback.print_tb(e.__traceback__, limit=99, file=sys.stdout)

            pass

        history_latents= None
        latent_paddings = None
        noise = None
        indices = None
        clean_latent_indices_pre = None
        blank_indices = None
        latent_indices = None
        clean_latent_indices_post = None
        clean_latent_2x_indices = None
        clean_latent_4x_indices = None
        clean_latent_indices = None
        clean_latents_pre = None
        clean_latents_post = None
        clean_latents_2x = None
        clean_latents_4x = None
        clean_latents = None
        clean_latent_indices_start = None
        clean_latent_1x_indices = None
        clean_latents_1x = None
        video_latent_batches = None
        video_image_batch = None
        latent_embeds = None

        self.transformer.to(self.offload_device)
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        if len(latent_batches_gend) > 0:
            output_images = self.decode_batches(latent_batches_gend, self.skip_frames, is_v2v=True)
        elif interrupted:
            temp_image = Image.new('RGB', (self.width, self.height), color = 'yellow')
            output_images = pil2tensorSwap(temp_image)
        else:
            temp_image = Image.new('RGB', (self.width, self.height), color = 'red')
            output_images = pil2tensorSwap(temp_image)

        output_images.to(device=self.offload_device)
        latent_batches_gend = None

        generation_status = "{}\nImages Generated: {}".format(generation_status, output_images.shape[0])

        if self.verbose_messaging:
            print("output_images Shape: {}".format(output_images.shape))

        return (output_images, generation_status, self.fps, )

    def generate_video(self, width, height, num_frames, seed_generator, noise_latent, llama_vec, llama_attention_mask, clip_l_pooler, llama_vec_n, llama_attention_mask_n, clip_l_pooler_n, image_encoder_last_hidden_state,
                        latent_indices, clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices, callback):
        with torch.autocast(device_type=mm.get_autocast_device(self.device), dtype=self.base_dtype, enabled=True):
            generated_latents = sample_hunyuan2(
                transformer=self.transformer,
                sampler=self.sampler,
                initial_latent=None,
                strength=self.denoise_strength,
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=self.cfg,
                distilled_guidance_scale=self.guidance_scale,
                guidance_rescale=0,
                shift=self.shift if Decimal(self.shift).compare(Decimal(0.0)) != 0 else None,
                num_inference_steps=self.steps,
                generator=seed_generator,
                noise=noise_latent,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=self.device,
                dtype=self.base_dtype,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

        return generated_latents

    def get_video_clean_latents(self, video_latents, context_frames, is_plus_one=True):
        from torch import Tensor
        latents_tuple = torch.split(video_latents, 1, dim=2)
        latents_split = [item for item in latents_tuple]
        latents_tuple = None

        if self.verbose_messaging:
            print("get_video_clean_latents: context_frames: {}".format(context_frames))
            print("get_video_clean_latents: video_latents.shape[2]: {}".format(video_latents.shape[2]))

        if context_frames < 3:
            context_frames = 3

        if context_frames >= video_latents.shape[2]:
            clean_latent_indices = []
            i = 0
            while i < video_latents.shape[2]:
                clean_latent_indices.append(i)

                i +=1
        elif context_frames > 3:
            if context_frames > (video_latents.shape[2] - 3):
                context_frames = video_latents.shape[2] - 3

            clean_latent_indices = []
            i = 0

            offset = math.ceil((video_latents.shape[2] - 2 - self.buffer_length) / (context_frames - 2))

            if offset < 1:
                offset = 1

            print("offset: {}".format(offset))
            index = 0
            while i < (context_frames - 2):
                if index < (video_latents.shape[2] - 1):
                    clean_latent_indices.append(int(index + self.buffer_length))

                index += offset
                i += 1

            clean_latent_indices.append(int(video_latents.shape[2] - 2))
            clean_latent_indices.append(int(video_latents.shape[2] - 1))

            clean_latents = None
            for index in clean_latent_indices:
                # print("get_video_clean_latents: index: {}".format(index))
                if not clean_latents is None:
                    clean_latents = torch.cat((clean_latents, latents_split[index]), 2)
                else:
                    clean_latents = latents_split[index]
        else:
            clean_latent_indices = [self.buffer_length, video_latents.shape[2] - 2, video_latents.shape[2] - 1]
            clean_latents = torch.cat([latents_split[self.buffer_length], latents_split[len(latents_split) - 2], latents_split[len(latents_split) - 1]], 2)

        return_indices = Tensor([clean_latent_indices])

        latent_indices = []

        i = self.buffer_length

        n = len(latents_split)

        while i < n:
            latent_indices.append(i)
            i += 1

        latent_indices = Tensor([latent_indices])
        latent_indices = latent_indices.to(dtype=torch.uint8, device=self.offload_device)

        latents_split = None

        return_indices = return_indices.to(latent_indices)

        return clean_latents, return_indices, latent_indices

    def setup_i2v_history_latents(self, history_latents, video_latents):
        cat_list = []

        cat_list.append(history_latents.to(dtype=torch.float32, device=self.offload_device))
        cat_list.append(video_latents.to(dtype=torch.float32, device=self.offload_device))

        history_latents = torch.cat(cat_list, dim=2)

        return history_latents

    def encoded_to_decoded_length(self, latent_length):
        if latent_length <= 0:
            return 0

        result_length = ((latent_length - 1) * 4) + 1

        return result_length

    def decoded_to_encoded_length(self, image_length):
        if image_length <= 0:
            return 0

        result_length = int(((image_length - 1) / 4) + 1)

        return result_length

    def get_latent_window_size(self, preferred_batch_size, frame_count, use_batch_size="next_lowest"):
        latent_size_factor = 4

        if self.verbose_messaging:
            print("get_latent_window_size: preferred_batch_size: {}".format(preferred_batch_size))
            print("get_latent_window_size: frame_count: {}".format(frame_count))
            print("get_latent_window_size: use_batch_size: {}".format(use_batch_size))

        num_frames = int(((frame_count - 1) // 4) * 4) + 1

        if num_frames != frame_count:
            print(f"Truncating video from {frame_count} to {num_frames} an because odd number of frames is not allowed.")

        if ((num_frames - 1) % (preferred_batch_size - 1)) == 0:
            print("(1) latent_window_size set to: {}".format(self.decoded_to_encoded_length(preferred_batch_size)))
            print("(1) batch_count set to: {}".format(int((num_frames - 1) / (preferred_batch_size - 1))))
            return self.decoded_to_encoded_length(preferred_batch_size), int((num_frames - 1) / (preferred_batch_size - 1)), num_frames

        if use_batch_size == "exact":
            num_frames_final = int(((num_frames - 1) // (preferred_batch_size - 1)) + 1) * (preferred_batch_size - 1)

            if num_frames_final != frame_count:
                print(f"Truncating video from {num_frames} to {num_frames_final} frames for preferred_batch_size compatibility.")

            print("(2) latent_window_size set to: {}".format(self.decoded_to_encoded_length(preferred_batch_size + 1)))
            print("(2) batch_count set to: {}".format(int((num_frames_final - 1) / (preferred_batch_size - 1))))
            return self.decoded_to_encoded_length(preferred_batch_size), int((num_frames_final - 1) / (preferred_batch_size - 1)), num_frames_final

        next_lowest_found = False
        next_highest_found = False
        next_lowest = preferred_batch_size - 1
        next_highest = preferred_batch_size - 1

        num_frames_final = int(((num_frames - 1) // 4) * 4) + 1

        if num_frames != num_frames_final:
            print(f"Truncating video from {num_frames} to {num_frames_final} frames for latent_window_size compatibility.")

        if (use_batch_size == "closest") or (use_batch_size == "next_lowest"):
            while next_lowest >= 12:
                next_lowest -= 4

                if (int((num_frames_final - 1) // 4) % next_lowest) == 0:
                    next_lowest_found = True
                    break

            next_lowest += 1

            if next_lowest_found and (use_batch_size == "next_lowest"):
                print("(3) latent_window_size set to: {}".format(self.decoded_to_encoded_length(next_lowest + 1)))
                print("(3) batch_count set to: {}".format(int((num_frames_final - 1) / next_lowest)))
                return self.decoded_to_encoded_length(next_lowest + 1), int((num_frames_final - 1) / next_lowest), num_frames_final

        while next_highest <= 156:
            next_highest += 4

            if (int((num_frames_final - 1) // 4) % next_highest) == 0:
                next_highest_found = True
                break

        if next_highest_found and (use_batch_size == "next_highest"):
            print("(4) latent_window_size set to: {}".format(self.decoded_to_encoded_length(next_highest + 1)))
            print("(4) batch_count set to: {}".format(int((num_frames_final - 1) / next_highest)))
            return self.decoded_to_encoded_length(next_highest + 1), int((num_frames_final - 1) / next_highest), num_frames_final

        if next_highest_found and next_lowest_found:
            if (preferred_batch_size - next_lowest) <= (next_highest - preferred_batch_size):
                print("(5) latent_window_size set to: {}".format(self.decoded_to_encoded_length(next_lowest + 1)))
                print("(5) batch_count set to: {}".format(int((num_frames_final - 1) / next_lowest)))
                return self.decoded_to_encoded_length(next_lowest + 1), int((num_frames_final - 1) / next_lowest), num_frames_final
            elif (next_highest - preferred_batch_size) < (preferred_batch_size - next_lowest):
                print("(6) latent_window_size set to: {}".format(self.decoded_to_encoded_length(next_highest + 1)))
                print("(6) batch_count set to: {}".format(int((num_frames_final - 1) / next_highest)))
                return self.decoded_to_encoded_length(next_highest + 1), int((num_frames_final - 1) / next_highest), num_frames_final

        raise ValueError("Unable to find a compatible latent_window_size for number of frames = {} and preferred_batch_size = {}.".format(frame_count, preferred_batch_size))

    def clip_vision_encode(self, image, crop="center"):
            crop_image = True
            if crop != "center":
                crop_image = False
            output = self.clip_vision.encode_image(image, crop=crop_image)
            return output

    def get_video_latent_embeds(self, video_frames, batch_frame_count, paddings):
        if len(video_frames.shape) < 4:
            video_frames = video_frames.unsqueeze(0)

        image_batches_tuple = torch.split(video_frames, batch_frame_count, dim=0)
        image_batches_split = [item for item in image_batches_tuple]
        image_batches_tuple = None

        latent_embeds = {}

        print("get_video_latent_embeds Batch: Encoding Start/End Images...")

        start_latent = None
        start_latent_embedding = None

        i = 0
        for batch_number in paddings:
            batch = image_batches_split[batch_number]

            batch_image_batches_tuple = torch.split(batch, 1, dim=0)
            batch_image_batches_split = [item for item in batch_image_batches_tuple]
            batch_image_batches_tuple = None

            latent_embeds["{}".format(batch_number)] = {"start_latent": None, "end_latent": None, "start_embedding": None, "end_embedding": None}

            if batch.shape[0] == batch_frame_count:
                start_index = batch.shape[0] - 1

                start_image = batch_image_batches_split[0].clone().detach().to(self.device)
                start_latent = self.encode_batched(start_image, self.latent_window_size)
                start_latent_embedding = self.clip_vision_encode(start_image)

                end_image = batch_image_batches_split[len(batch_image_batches_split) - 1].clone().detach().to(self.device)
                end_latent = self.encode_batched(end_image, self.latent_window_size)
                end_latent_embedding = self.clip_vision_encode(end_image)

                latent_embeds["{}".format(batch_number)]["start_latent"] = start_latent["samples"] * vae_scaling_factor
                latent_embeds["{}".format(batch_number)]["start_embedding"] = start_latent_embedding
                latent_embeds["{}".format(batch_number)]["end_latent"] = end_latent["samples"] * vae_scaling_factor
                latent_embeds["{}".format(batch_number)]["end_embedding"] = end_latent_embedding

                start_image = None
                start_latent = None
                start_latent_embedding = None
                end_image = None
                end_latent = None
                end_latent_embedding = None
                batch_image_batches_split = None

            i += 1

        print("get_video_latent_embeds Batch: Encoding Start/End Images...Done")

        mm.soft_empty_cache()
        gc.collect()
        time.sleep(0.2)

        return latent_embeds

    def encode_tiled(self, images):
        if len(images.shape) < 4:
            images = images.unsqueze(0)

        encoded_data = partial_encode_tiled(self.vae, images)

        if len(encoded_data.shape) < 5:
            encoded_data.unsqueeze(0)

        return encoded_data
    def decode_tiled(self, latents, unload_after=True):
        decoded_data = partial_decode_tiled(self.vae, latents, self.dec_tile_size, self.dec_overlap, self.dec_temporal_size, self.dec_temporal_overlap, unload_after)

        if len(decoded_data.shape) < 4:
            decoded_data.unsqueeze(0)

        return decoded_data
    def decode_batches(self, latent_batches, skip_frames, is_v2v=False):
        if (latent_batches is None) or (len(latent_batches) < 1):
            print("decode_batches: Warning...nothing to decode.")
            return None

        mm.unload_all_models()
        gc.collect()
        time.sleep(1)

        resulting_images = None

        if self.blend_frames < 1:
            i = 0
            for entry in latent_batches:
                entry.to(dtype=torch.float32, device=self.device)

                if i < (len(latent_batches) - 1):
                    temp_decoded = self.decode_tiled(entry, unload_after=False)
                else:
                    temp_decoded = self.decode_tiled(entry, unload_after=True)

                if len(temp_decoded.shape) < 4:
                    temp_decoded = temp_decoded.unsqueeze(0)

                if is_v2v:
                    image_batches_tuple = torch.split(temp_decoded, temp_decoded.shape[0] - 3, dim=0)
                    image_batches_split = [item for item in image_batches_tuple]
                    temp_decoded = image_batches_split[0]
                    image_batches_tuple = None
                    image_batches_split = None

                if not resulting_images is None:
                    resulting_images = torch.cat((resulting_images, temp_decoded), 0)
                else:
                    resulting_images = temp_decoded

                entry.to(device=self.offload_device)

                mm.soft_empty_cache()
                gc.collect()
                time.sleep(0.2)

                i += 1
        else:
            temp_decoded_batches = []

            i = 0
            for entry in latent_batches:
                entry.to(dtype=torch.float32, device=self.device)

                if i < (len(latent_batches) - 1):
                    temp_decoded = self.decode_tiled(entry, unload_after=False)
                else:
                    temp_decoded = self.decode_tiled(entry, unload_after=True)

                if len(temp_decoded.shape) < 4:
                    temp_decoded = temp_decoded.unsqueeze(0)

                if is_v2v:
                    image_batches_tuple = torch.split(temp_decoded, temp_decoded.shape[0] - 3, dim=0)
                    image_batches_split = [item for item in image_batches_tuple]
                    temp_decoded = image_batches_split[0]
                    image_batches_tuple = None
                    image_batches_split = None

                temp_decoded_batches.append(temp_decoded)

                entry.to(device=self.offload_device)

                mm.soft_empty_cache()
                gc.collect()
                time.sleep(0.2)

                i += 1

            blend_value = 1.0 / self.blend_frames
            i = 0
            while i < (len(temp_decoded_batches) - 1):
                alpha_blend_val = blend_value
                blend_count = self.blend_frames

                image_batches_tuple = torch.split(temp_decoded_batches[i], 1, dim=0)
                image_batches_split = [item for item in image_batches_tuple]
                image1 = image_batches_split[len(image_batches_split) - 1]
                image_batches_tuple = None
                image_batches_split = None

                image_batches_tuple = torch.split(temp_decoded_batches[i + 1], 1, dim=0)
                image_batches_split = [item for item in image_batches_tuple]
                image2 = image_batches_split[0]
                image_batches_tuple = None
                image_batches_split = None

                image1 = tensor2pilSwap(image1)[0]
                image2 = tensor2pilSwap(image2)[0]

                blend_latents = None

                while blend_count > 0:
                    blended_image = Image.blend(image1, image2, alpha_blend_val)
                    temp_latent = pil2tensorSwap(blended_image)
                    blended_image = None

                    if len(temp_latent.shape) < 4:
                        temp_latent = temp_latent.unsqueeze(0)

                    if not blend_latents is None:
                        blend_latents = torch.cat((blend_latents, temp_latent), 0)
                    else:
                        blend_latents = temp_latent

                    alpha_blend_val += blend_value
                    blend_count -= 1

                temp_decoded_batches[i] = torch.cat((temp_decoded_batches[i], blend_latents), 0)
                blend_latents = None

                mm.soft_empty_cache()
                gc.collect()
                time.sleep(0.2)

                i += 1

            for entry in temp_decoded_batches:
                if not resulting_images is None:
                    resulting_images = torch.cat((resulting_images, entry), 0)
                else:
                    resulting_images = entry

                entry.to(device=self.offload_device)

            temp_decoded_batches = None
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)


        print("decode_batches: Full decoded images count: {}".format(resulting_images.shape[0]))

        if skip_frames < 1:
            return resulting_images

        skipped_frames = 1

        image_batches_tuple = torch.split(resulting_images, 1, dim=0)
        image_batches_split = [item for item in image_batches_tuple]

        resulting_images = None

        for image in image_batches_split:
            if skipped_frames <= skip_frames:
                skipped_frames += 1
                continue

            if not resulting_images is None:
                resulting_images = torch.cat((resulting_images, image), 0)
            else:
                resulting_images = image

        print("decode_batches: Final decoded images count: {}".format(resulting_images.shape[0]))

        return resulting_images

    def encode_batched(self, image, per_batch=16):
        from nodes import VAEEncode
        from comfy.utils import ProgressBar

        t = []
        pbar = ProgressBar(image.shape[0])
        for start_idx in range(0, image.shape[0], per_batch):
            try:
                sub_pixels = self.vae.vae_encode_crop_pixels(image[start_idx:start_idx+per_batch])
            except:
                sub_pixels = VAEEncode.vae_encode_crop_pixels(image[start_idx:start_idx+per_batch])
            t.append(self.vae.encode(sub_pixels[:,:,:,:3]))
            pbar.update(per_batch)

        return {"samples": torch.cat(t, dim=0)}

    def video_encode(self, video_frames, batch_size):
        if len(video_frames.shape) < 4:
            video_frames = video_frames.unsqueeze(0)

        image_batches_tuple = torch.split(video_frames, batch_size - 1, dim=0)
        image_batches_split = [item for item in image_batches_tuple]

        final_image_batches = []
        last_frame = None

        for entry in image_batches_split:
            if entry.shape[0] == (batch_size - 1):
                final_image_batches.append(entry.clone().detach())
            else:
                if last_frame is None:
                    last_frame = entry

        print(f"video_encode: Encoding input video frames in batch size {batch_size} (reduce preferred_batch_size if memory issues here or if forcing video resolution)")
        latents = None
        final_latent_batches = []
        batch_encodings = []

        i = 1
        with torch.no_grad():
            for entry in final_image_batches:
                entry.to(dtype=torch.float32, device=self.device)

                if (i < len(final_image_batches)):
                    next_start_image = final_image_batches[i][1:2, :, :, :]

                    entry = torch.cat((entry, next_start_image.to(entry)), 0)
                    batch_latent = partial_encode_tiled(self.vae, entry, unload_after=False)
                else:
                    entry = torch.cat((entry, last_frame.to(entry)), 0)
                    batch_latent = partial_encode_tiled(self.vae, entry, unload_after=True)

                i += 1

                if len(batch_latent.shape) < 5:
                    batch_latent = batch_latent.unsqueeze(0)

                batch_latent = batch_latent * vae_scaling_factor
                final_latent_batches.append(batch_latent.clone().detach())
                entry.to(device=self.offload_device)

        print(f"Encoding input video frames in batch size {batch_size} Done.")

        return final_latent_batches, batch_encodings

class WarpedFramepackLoraSelectBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "fuse_lora": ("BOOLEAN", {"default": False, "tooltip": "Fuse the LORA model with the base model. This is recommended for better performance."}),
            },
            "optional": {
                "prev_lora":("FPLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
            }
        }

    RETURN_TYPES = ("FPLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "Warped/Framepack/Lora"
    DESCRIPTION = "Select a LoRA model from ComfyUI/models/loras"

    def getlorapath(self, lora, strength, prev_lora=None, fuse_lora=True):
        loras_list = []

        lora = {
            "path": folder_paths.get_full_path("loras", lora),
            "strength": strength,
            "name": lora.split(".")[0],
            "fuse_lora": fuse_lora,
        }
        if prev_lora is not None:
            loras_list.extend(prev_lora)

        loras_list.append(lora)
        return (loras_list,)
