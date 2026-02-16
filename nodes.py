import nodes
import torch
import comfy.model_management as mm
from PIL import ImageDraw, Image, ImageOps, ImageChops, ImageColor
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

from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from enum import Enum

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
import io

import node_helpers

from torch import Tensor
from einops import repeat
from unittest.mock import patch

from comfy.ldm.flux.layers import timestep_embedding, apply_mod
from comfy.ldm.lightricks.model import precompute_freqs_cis
from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
from comfy.ldm.common_dit import rms_norm
from comfy.ldm.wan.model import sinusoidal_embedding_1d

from .diffusers_helper.memory import move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation
from .diffusers_helper.k_diffusion_hunyuan import sample_hunyuan, sample_hunyuan2
from .diffusers_helper.utils import crop_or_pad_yield_mask, soft_append_bcthw
import math

import threading
import copy

import safetensors
import einops
import socket
import re

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.tiff', '.gif', '.bmp', '.webp', '.avif', '.vfif')
ALLOWED_VIDEO_EXT = ('mp4', 'flv', 'mov', 'avi', 'mpg', 'webm', 'mkv')
ALLOWED_CAPTION_EXT = ('.txt')

vae_scaling_factor = 0.476986

class VideoGenerationType(Enum):
    T2V = "t2v"
    I2V = "i2v"
    V2V = "v2v"

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

        return new_tensor

    if device is None:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    new_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).to(device).unsqueeze(0)

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

def get_default_gen_output_path():
    default_path = "{}\\MyLoras\\new_lora_hy.safetensors".format(folder_paths.output_directory)
    return default_path

def get_default_pt_st_output_folder():
    default_path = "{}\\pt_conversions".format(folder_paths.output_directory)
    return default_path

def get_default_output_path():
    default_path = "{}\\MergedHunyuanLoras\\new_lora_hy.safetensors".format(folder_paths.output_directory)
    return default_path

def get_default_output_folder():
    default_folder = "{}\\MergedHunyuanLoras".format(folder_paths.output_directory)
    return default_folder

def get_default_xl_output_folder():
    default_folder = "{}\\MergedXLLoras".format(folder_paths.output_directory)
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

def convert_key_format(key: str) -> str:
    """Standardize LoRA key format by removing prefixes."""
    prefixes = ["diffusion_model.", "transformer."]
    for prefix in prefixes:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    return key

def filter_lora_keys(lora: Dict[str, torch.Tensor], blocks_type: str) -> Dict[str, torch.Tensor]:
    """Filter LoRA weights based on block type."""
    if blocks_type == "all":
        return lora

    filtered_lora = {}

    for key, value in lora.items():
        base_key = convert_key_format(key)
        if blocks_type in base_key:
            filtered_lora[key] = value
    return filtered_lora

def convert_to_musubi(lora: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Checks for and converts to Musubi Tuner format which supports Network Alpha and uses different naming."""
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
        print("Lora is already in musubi format. Nothing to convert.")
        return lora

    print("Converting lora to musubi format.")

    temp_lora = convert_lora(lora, do_check_for_musuibi=False)

    for key in temp_lora.keys():
        print("Intermediate Lora Key: {}".format(key))

    print("\n")

    converted_lora = {}
    prefix = "diffusion_model."
    has_double_blocks = False
    has_single_blocks = False

    for key, weight in temp_lora.items():
        print("Converting Key: {}".format(key))

        if "linear" in key:
            print("Skipping Key.")
            continue

        if ("double.blocks" in key) or ("double_blocks" in key):
            has_double_blocks = True

        elif ("single.blocks" in key) or ("single_blocks" in key):
            has_single_blocks = True

        # lora_name = key.split("_", 1)[0]  # before first dot

        # HunyuanVideo lora name to module name: ugly but works
        # module_name = lora_name[len(prefix) :]  # remove "diffusion_model."

        module_name = key.replace(prefix, "")

        print("module_name: {}".format(module_name))

        module_name = module_name.replace(".", "_")  # replace "_" with "."
        module_name = module_name.replace("double.blocks.", "double_blocks_")  # fix double blocks
        module_name = module_name.replace("single.blocks.", "single_blocks_")  # fix single blocks
        module_name = module_name.replace("img.", "img_")  # fix img
        module_name = module_name.replace("txt.", "txt_")  # fix txt
        module_name = module_name.replace("attn.", "attn_")  # fix attn

        musubi_prefix = "lora_unet"

        if "lora_A" in key:
            new_key = f"{musubi_prefix}_{module_name.replace('_lora_A_weight', '.lora_down.weight')}"
        elif "lora_B" in key:
            new_key = f"{musubi_prefix}_{module_name.replace('_lora_B_weight', '.lora_up.weight')}"
        else:
            print(f"unexpected key: {key} in diffusion_model LoRA format")
            continue

        print("New Key: {}".format(new_key))

        converted_lora[new_key] = weight.to(dtype=torch.bfloat16)

    if has_double_blocks:
        print("Lora Has Double Blocks.")

        for block_num in range(20):
            converted_lora[f"lora_unet_double_blocks_{block_num}_img_attn_proj.alpha"] = torch.empty([]).to(dtype=torch.bfloat16)
            converted_lora[f"lora_unet_double_blocks_{block_num}_img_attn_qkv.alpha"] = torch.empty([]).to(dtype=torch.bfloat16)
            converted_lora[f"lora_unet_double_blocks_{block_num}_img_mlp_fc1.alpha"] = torch.empty([]).to(dtype=torch.bfloat16)
            converted_lora[f"lora_unet_double_blocks_{block_num}_img_mlp_fc2.alpha"] = torch.empty([]).to(dtype=torch.bfloat16)
            converted_lora[f"lora_unet_double_blocks_{block_num}_txt_attn_proj.alpha"] = torch.empty([]).to(dtype=torch.bfloat16)
            converted_lora[f"lora_unet_double_blocks_{block_num}_txt_attn_qkv.alpha"] = torch.empty([]).to(dtype=torch.bfloat16)
            converted_lora[f"lora_unet_double_blocks_{block_num}_txt_mlp_fc1.alpha"] = torch.empty([]).to(dtype=torch.bfloat16)
            converted_lora[f"lora_unet_double_blocks_{block_num}_txt_mlp_fc2.alpha"] = torch.empty([]).to(dtype=torch.bfloat16)

    if has_single_blocks:
        print("Lora Has Single Blocks.")

        for block_num in range(40):
            converted_lora[f"lora_unet_single_blocks_{block_num}_linear1.alpha"] = torch.Tensor([]).to(dtype=torch.bfloat16)
            converted_lora[f"lora_unet_single_blocks_{block_num}_linear2.alpha"] = torch.Tensor([]).to(dtype=torch.bfloat16)

    return converted_lora

def check_for_musubi(lora: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

def convert_lora(lora, convert_to="diffusion_model", do_check_for_musuibi=True):
    if do_check_for_musuibi:
        temp_lora = check_for_musubi(lora)
    else:
        temp_lora = lora

    new_lora = {}

    for key in temp_lora.keys():
        if convert_to in key:
            new_lora = temp_lora
            break

        if key.startswith("transformer.") and (convert_to == "diffusion_model"):
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

    return new_lora

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

def convert_lora_dimensions(max_dimension, lora):
    new_lora = {}

    for key in lora.keys():
        temp_weights = lora[key]

        # if (temp_weights.shape[0] == max_dimension) or (temp_weights.shape[1] == max_dimension):
        #     return lora

        if temp_weights.shape[0] < temp_weights.shape[1]:
            padding = torch.zeros([max_dimension, temp_weights.shape[1]])

            if temp_weights.shape[0] <= max_dimension:
                padding[:temp_weights.shape[0],:] = temp_weights
                new_lora[key] = padding
            else:
                padding[:max_dimension,:] = temp_weights[:max_dimension,:]
                new_lora[key] = padding
        else:
            padding = torch.zeros([temp_weights.shape[0], max_dimension])

            if temp_weights.shape[1] <= max_dimension:
                padding[:,:temp_weights.shape[1]] = temp_weights
                new_lora[key] = padding
            else:
                padding[:,:max_dimension] = temp_weights[:,:max_dimension]
                new_lora[key] = padding
    lora = None

    return new_lora

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
    CATEGORY = "Warped/Hunyuan/Merge"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

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

    def merge_multiple_loras(self, save_path, lora_1, strength_1, blocks_type_1, lora_2, strength_2, blocks_type_2, save_metadata=True):
        """Load and apply multiple LoRA models."""
        temp_loras = {}
        metadata = {"loras": "{} and {}".format(lora_1, lora_2)}
        metadata["strengths"] = "{} and {}".format(strength_1, strength_2)
        metadata["block_types"] = "{} and {}".format(blocks_type_1, blocks_type_2)

        if lora_1 != "None" and strength_1 != 0:
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_1, 1.0, "all")
            lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
            filtered_lora = filter_lora_keys(lora_weights, blocks_type_1)
            temp_loras["1"] = {"lora_weights": lora_weights, "strength": strength_1, "filtered_lora": filtered_lora}

        if lora_2 != "None" and strength_2 != 0:
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_2, 1.0, "all")
            lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
            filtered_lora = filter_lora_keys(lora_weights, blocks_type_2)
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
    CATEGORY = "Warped/Hunyuan/Merge"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

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

    def merge_multiple_loras(self, save_path, lora_1, strength_1, blocks_type_1, lora_2, strength_2, blocks_type_2, lora_3, strength_3, blocks_type_3, lora_4, strength_4, blocks_type_4, save_metadata=True):
        temp_loras = {}
        metadata = {"loras": "{} and {} and {} and {}".format(lora_1, lora_2, lora_3, lora_4)}
        metadata["strengths"] = "{} and {} and {} and {}".format(strength_1, strength_2, strength_3, strength_4)
        metadata["block_types"] = "{} and {} and {} and {}".format(blocks_type_1, blocks_type_2, blocks_type_3, blocks_type_4)

        if lora_1 != "None" and strength_1 != 0:
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_1, 1.0, "all")
            lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
            filtered_lora = filter_lora_keys(lora_weights, blocks_type_1)
            temp_loras["1"] = {"lora_weights": lora_weights, "strength": strength_1, "filtered_lora": filtered_lora}

        if lora_2 != "None" and strength_2 != 0:
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_2, 1.0, "all")
            lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
            filtered_lora = filter_lora_keys(lora_weights, blocks_type_2)
            temp_loras["2"] = {"lora_weights": lora_weights, "strength": strength_2, "filtered_lora": filtered_lora}

        if lora_3 != "None" and strength_3 != 0:
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_3, 1.0, "all")
            lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
            filtered_lora = filter_lora_keys(lora_weights, blocks_type_3)
            temp_loras["3"] = {"lora_weights": lora_weights, "strength": strength_3, "filtered_lora": filtered_lora}

        if lora_4 != "None" and strength_4 != 0:
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_4, 1.0, "all")
            lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
            filtered_lora = filter_lora_keys(lora_weights, blocks_type_4)
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
                                temp_weights = padding.to(dtype=torch.bfloat16)
                            elif temp_weights.shape[0] > new_lora[key].shape[0]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([temp_weights.shape[0], temp_weights.shape[1]])
                                padding[:new_lora[key].shape[0],:] = new_lora[key]
                                new_lora[key] = padding.to(dtype=torch.bfloat16)
                        else:
                            if temp_weights.shape[1] < new_lora[key].shape[1]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([new_lora[key].shape[0], new_lora[key].shape[1]])
                                padding[:,:temp_weights.shape[1]] = temp_weights
                                temp_weights = padding.to(dtype=torch.bfloat16)
                            elif temp_weights.shape[1] > new_lora[key].shape[1]:
                                temp_weights = temp_weights.clone().detach()
                                new_lora[key] = new_lora[key].clone().detach()

                                padding = torch.zeros([temp_weights.shape[0], temp_weights.shape[1]])
                                padding[:,:new_lora[key].shape[1]] = new_lora[key]
                                new_lora[key] = padding.to(dtype=torch.bfloat16)

                        try:
                            new_lora[key] = torch.add(new_lora[key], temp_weights).to(dtype=torch.bfloat16)
                        except Exception as e:
                            raise(e)
                    else:
                        new_lora[key] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"]).to(dtype=torch.bfloat16)

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
                "blocks_type_1": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_2": (['None'] + get_lora_list(),),
                "blocks_type_2": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "merge_multiple_loras"
    CATEGORY = "Warped/Hunyuan/Merge"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

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

    def merge_multiple_loras(self, save_path, lora_1, blocks_type_1, lora_2, blocks_type_2, save_metadata=True):
        """Load and apply multiple LoRA models."""
        strength = 1.0000
        temp_loras = {}
        metadata = {"loras": "{} and {}".format(lora_1, lora_2)}
        metadata["strengths"] = "{} and {}".format(strength, strength)
        metadata["block_types"] = "{} and {}".format(blocks_type_1, blocks_type_2)

        if lora_1 != "None":
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_1, 1.0, "all")
            lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
            filtered_lora = filter_lora_keys(lora_weights, blocks_type_1)
            temp_loras["1"] = {"lora_weights": lora_weights, "strength": strength, "filtered_lora": filtered_lora}

        if lora_2 != "None":
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_2, 1.0, "all")
            lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
            filtered_lora = filter_lora_keys(lora_weights, blocks_type_2)
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
    CATEGORY = "Warped/Hunyuan/Merge"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

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

    def merge_multiple_loras(self, save_path, lora_1, blocks_type_1, lora_2, blocks_type_2, lora_3, blocks_type_3, lora_4, blocks_type_4, save_metadata=True):
        strength = 1.0000
        temp_loras = {}
        metadata = {"loras": "{} and {} and {} and {}".format(lora_1, lora_2, lora_3, lora_4)}
        metadata["strengths"] = "{} and {} and {} and {}".format(strength, strength, strength, strength)
        metadata["block_types"] = "{} and {} and {} and {}".format(blocks_type_1, blocks_type_2, blocks_type_3, blocks_type_4)

        if lora_1 != "None":
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_1, 1.0, "all")
            lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
            filtered_lora = filter_lora_keys(lora_weights, blocks_type_1)
            temp_loras["1"] = {"lora_weights": lora_weights, "strength": strength, "filtered_lora": filtered_lora}

        if lora_2 != "None":
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_2, 1.0, "all")
            lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
            filtered_lora = filter_lora_keys(lora_weights, blocks_type_2)
            temp_loras["2"] = {"lora_weights": lora_weights, "strength": strength, "filtered_lora": filtered_lora}

        if lora_3 != "None":
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_3, 1.0, "all")
            lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
            filtered_lora = filter_lora_keys(lora_weights, blocks_type_3)
            temp_loras["3"] = {"lora_weights": lora_weights, "strength": strength, "filtered_lora": filtered_lora}

        if lora_4 != "None":
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_4, 1.0, "all")
            lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
            filtered_lora = filter_lora_keys(lora_weights, blocks_type_4)
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

# class WarpedHunyuanMultiLoraMixer:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "save_folder": ("STRING", {"default": get_default_output_folder()}),
#                 "model_prefix": ("STRING", {"default": "new_model_hy"}),
#                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
#                 "num_output": ("INT", {"default": 1, "min": 1, "max": 100}),
#                 "lora_1": (['None'] + get_lora_list(),),
#                 "lora_2": (['None'] + get_lora_list(),),
#                 "lora_3": (['None'] + get_lora_list(),),
#                 "lora_4": (['None'] + get_lora_list(),),
#                 "lora_5": (['None'] + get_lora_list(),),
#                 "lora_6": (['None'] + get_lora_list(),),
#                 "lora_7": (['None'] + get_lora_list(),),
#                 "lora_8": (['None'] + get_lora_list(),),
#                 "save_metadata": ("BOOLEAN", {"default": True}),
#             },
#         }
#
#     RETURN_TYPES = ()
#     OUTPUT_NODE = True
#     OUTPUT_IS_LIST = (True,)
#     FUNCTION = "merge_multiple_loras"
#     CATEGORY = "Warped/Hunyuan/Mixers"
#     DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."
#
#     def convert_key_format(self, key: str) -> str:
#         """Standardize LoRA key format by removing prefixes."""
#         prefixes = ["diffusion_model.", "transformer."]
#         for prefix in prefixes:
#             if key.startswith(prefix):
#                 key = key[len(prefix):]
#                 break
#         return key
#
#     def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
#         """Load and filter a single LoRA model."""
#         if not lora_name or strength == 0:
#             return {}, {}
#
#         # Get the full path to the LoRA file
#         lora_path = folder_paths.get_full_path("loras", lora_name)
#         if not os.path.exists(lora_path):
#             raise ValueError(f"LoRA file not found: {lora_path}")
#
#         # Load the LoRA weights
#         lora_weights = utils.load_torch_file(lora_path)
#
#         return lora_weights
#
#     def get_mixtures(self, seed, num_output, lora_keys):
#         random.seed(seed)
#         mixtures = {}
#
#         for i in range(num_output):
#             mixtures["{}".format(i + 1)] = {}
#
#         for key in lora_keys:
#             for mixture_key in mixtures.keys():
#                 mixtures[mixture_key][key] = {"single": [], "double": []}
#
#         for mixture_key in mixtures.keys():
#             for j in range(40):
#                 temp_key = "{}".format(random.randint(1, len(lora_keys)))
#                 mixtures[mixture_key][temp_key]["single"].append(j)
#
#             for j in range(20):
#                 temp_key = "{}".format(random.randint(1, len(lora_keys)))
#                 mixtures[mixture_key][temp_key]["double"].append(j)
#
#             i += 1
#
#         print("\nMixtures\n")
#
#         block_metadata = ""
#
#         for mixture_key in mixtures.keys():
#             for key in mixtures[mixture_key]:
#                 print("{} | {}: {}".format(mixture_key, key, mixtures[mixture_key][key]))
#
#                 if len(block_metadata) > 0:
#                     block_metadata = "{}  |  {}".format(block_metadata, "{}: {}: {}".format(mixture_key, key, mixtures[mixture_key][key]))
#                 else:
#                     block_metadata = "{}: {}: {}".format(mixture_key, key, mixtures[mixture_key][key])
#
#             print("\n")
#
#         return mixtures, block_metadata
#
#     def merge_multiple_loras(self, save_folder, model_prefix, seed, num_output, lora_1, lora_2, lora_3, lora_4, lora_5, lora_6, lora_7, lora_8, save_metadata=True):
#         print("Save_folder: {}".format(save_folder))
#         os.makedirs(save_folder, exist_ok = True)
#
#         temp_loras = {}
#         metadata = {"loras": "{} and {} and {} and {} and {} and {} and {} and {}".format(lora_1, lora_2, lora_3, lora_4, lora_5, lora_6, lora_7, lora_8)}
#         metadata["seed"] = "{}".format(seed)
#         metadata["num_output"] = "{}".format(num_output)
#
#         if lora_1 != "None":
#             print(lora_1)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_1, 1.0, "all")
#             lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
#             temp_loras["1"] = {"lora_weights": lora_weights}
#
#         if lora_2 != "None":
#             print(lora_2)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_2, 1.0, "all")
#             lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
#             temp_loras["2"] = {"lora_weights": lora_weights}
#
#         if lora_3 != "None":
#             print(lora_3)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_3, 1.0, "all")
#             lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
#             temp_loras["3"] = {"lora_weights": lora_weights}
#
#         if lora_4 != "None":
#             print(lora_4)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_4, 1.0, "all")
#             lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
#             temp_loras["4"] = {"lora_weights": lora_weights}
#
#         if lora_5 != "None":
#             print(lora_5)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_5, 1.0, "all")
#             lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
#             temp_loras["5"] = {"lora_weights": lora_weights}
#
#         if lora_6 != "None":
#             print(lora_6)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_6, 1.0, "all")
#             lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
#             temp_loras["6"] = {"lora_weights": lora_weights}
#
#         if lora_7 != "None":
#             print(lora_7)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_7, 1.0, "all")
#             lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
#             temp_loras["7"] = {"lora_weights": lora_weights}
#
#         if lora_8 != "None":
#             print(lora_8)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_8, 1.0, "all")
#             lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
#             temp_loras["8"] = {"lora_weights": lora_weights}
#
#         loras = {}
#         max_dimension = 0
#
#         for lora_key in temp_loras.keys():
#             # print(lora_key)
#             loras[lora_key] = {"lora_weights": {}}
#
#             for key in temp_loras[lora_key]["lora_weights"].keys():
#                 new_key = key.replace("transformer.", "diffusion_model.")
#                 loras[lora_key]["lora_weights"][new_key] = temp_loras[lora_key]["lora_weights"][key]
#
#                 temp_dimension = min(loras[lora_key]["lora_weights"][new_key].shape[0], loras[lora_key]["lora_weights"][new_key].shape[1])
#
#                 if temp_dimension > max_dimension:
#                     max_dimension = temp_dimension
#
#         merge_mixtures, block_metadata = self.get_mixtures(seed, num_output, loras.keys())
#         metadata["max_dimension"] = "{}".format(max_dimension)
#
#         print("Max Dimension: {}".format(max_dimension))
#
#         # convert the rank/dims for each lora to be mixed
#         for lora_key in temp_loras.keys():
#             temp_loras[lora_key]["lora_weights"] = convert_lora_dimensions(max_dimension, temp_loras[lora_key]["lora_weights"])
#
#         save_message = ""
#
#         for mixture_key in merge_mixtures:
#             new_lora = {}
#             output_filename = os.path.join(save_folder, "{}_{:05}.safetensors".format(model_prefix, int(mixture_key)))
#
#             metadata["merge_mixture"] = "{}".format(merge_mixtures[mixture_key])
#             # metadata["block_metadata"] = "{}".format(block_metadata[int(mixture_key)])
#
#             for lora_key in loras.keys():
#                 mixture_single_blocks = merge_mixtures[mixture_key][lora_key]["single"]
#                 mixture_double_blocks = merge_mixtures[mixture_key][lora_key]["double"]
#
#                 for key in loras[lora_key]["lora_weights"].keys():
#                     temp_strings = str(key).split('.')
#                     temp_block_num = int(temp_strings[2])
#
#                     if temp_strings[1] == "single_blocks":
#                         if temp_block_num in mixture_single_blocks:
#                             new_lora[key] = loras[lora_key]["lora_weights"][key].to(dtype=torch.bfloat16)
#                         continue
#
#                     if temp_strings[1] == "double_blocks":
#                         if temp_block_num in mixture_double_blocks:
#                             new_lora[key] = loras[lora_key]["lora_weights"][key].to(dtype=torch.bfloat16)
#
#             if not save_metadata:
#                 metadata = None
#
#             print("Saving Model To: {}...".format(output_filename))
#             utils.save_torch_file(new_lora, output_filename, metadata=metadata)
#             print("Saving Model To: {}...Done.".format(output_filename))
#
#             save_message = "{}\n{}".format(save_message, "Weights Saved To: {}".format(output_filename))
#
#             new_lora = None
#             mm.soft_empty_cache()
#             gc.collect()
#             time.sleep(1)
#
#         return {"ui": {"tags": ["save_message"]}}
#
# class WarpedHunyuanMultiLoraMixerExt:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "save_folder": ("STRING", {"default": get_default_output_folder()}),
#                 "model_prefix": ("STRING", {"default": "new_model_hy"}),
#                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
#                 "num_output": ("INT", {"default": 1, "min": 1, "max": 100}),
#                 "lora_1": (['None'] + get_lora_list(),),
#                 "strength_1": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "lora_2": (['None'] + get_lora_list(),),
#                 "strength_2": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "lora_3": (['None'] + get_lora_list(),),
#                 "strength_3": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "lora_4": (['None'] + get_lora_list(),),
#                 "strength_4": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "lora_5": (['None'] + get_lora_list(),),
#                 "strength_5": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "lora_6": (['None'] + get_lora_list(),),
#                 "strength_6": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "lora_7": (['None'] + get_lora_list(),),
#                 "strength_7": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "lora_8": (['None'] + get_lora_list(),),
#                 "strength_8": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "save_metadata": ("BOOLEAN", {"default": True}),
#                 "convert_to": (["diffusion_model", "transformer"], {"default": "diffusion_model"}),
#                 "max_dimension": ([32, 64, 128], {"default": 64}),
#             },
#         }
#
#     RETURN_TYPES = ()
#     OUTPUT_NODE = True
#     OUTPUT_IS_LIST = (True,)
#     FUNCTION = "merge_multiple_loras"
#     CATEGORY = "Warped/Hunyuan/Mixers"
#     DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."
#
#     def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
#         """Load and filter a single LoRA model."""
#         if not lora_name or strength == 0:
#             return {}, {}
#
#         # Get the full path to the LoRA file
#         lora_path = folder_paths.get_full_path("loras", lora_name)
#         if not os.path.exists(lora_path):
#             raise ValueError(f"LoRA file not found: {lora_path}")
#
#         # Load the LoRA weights
#         lora_weights = utils.load_torch_file(lora_path)
#
#         lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
#
#         return lora_weights
#
#     def get_random_key(self, keys):
#         if len(keys) > 0:
#             random_key = random.randint(0, len(keys) - 1)
#             # print("Keys: {}  |  Random Key: {}".format(random_key))
#             return keys[random_key]
#
#         return -1
#
#     def get_mixtures(self, seed, num_output, lora_keys, block_types):
#         random.seed(seed)
#         mixtures = {}
#
#         for i in range(num_output):
#             mixtures["{}".format(i + 1)] = {}
#
#         single_block_loras = []
#         double_block_loras = []
#
#         for key in lora_keys:
#             for mixture_key in mixtures.keys():
#                 mixtures[mixture_key][key] = {"single": [], "double": []}
#
#                 if block_types[key]["has_single_blocks"]:
#                     single_block_loras.append(int(key))
#
#                 if block_types[key]["has_double_blocks"]:
#                     double_block_loras.append(int(key))
#
#         for mixture_key in mixtures.keys():
#             if len(single_block_loras) > 0:
#                 for j in range(40):
#                     random_key = self.get_random_key(single_block_loras)
#                     temp_key = "{}".format(random_key)
#                     mixtures[mixture_key][temp_key]["single"].append(j)
#
#             if len(double_block_loras) > 0:
#                 for j in range(20):
#                     random_key = self.get_random_key(double_block_loras)
#                     temp_key = "{}".format(random_key)
#                     mixtures[mixture_key][temp_key]["double"].append(j)
#
#             i += 1
#
#         print("\nMixtures\n")
#
#         block_metadata = ""
#
#         for mixture_key in mixtures.keys():
#             for key in mixtures[mixture_key]:
#                 print("{} | {}: {}".format(mixture_key, key, mixtures[mixture_key][key]))
#
#                 if len(block_metadata) > 0:
#                     block_metadata = "{}  |  {}".format(block_metadata, "{}: {}: {}".format(mixture_key, key, mixtures[mixture_key][key]))
#                 else:
#                     block_metadata = "{}: {}: {}".format(mixture_key, key, mixtures[mixture_key][key])
#
#             print("\n")
#
#         return mixtures, block_metadata
#
#     def determine_lora_block_types(self, loras):
#         block_types = {}
#
#         for lora_key in loras.keys():
#             block_types[lora_key] = { "has_single_blocks": False, "has_double_blocks": False }
#
#             for key in loras[lora_key]["lora_weights"].keys():
#                 if "single_blocks" in key:
#                     block_types[lora_key]["has_single_blocks"] = True
#                 elif "double_blocks" in key:
#                     block_types[lora_key]["has_double_blocks"] = True
#
#                 if block_types[lora_key]["has_single_blocks"] and block_types[lora_key]["has_double_blocks"]:
#                     break
#
#         return block_types
#
#     def merge_multiple_loras(self, save_folder, model_prefix, seed, num_output, lora_1, strength_1, lora_2, strength_2, lora_3, strength_3, lora_4, strength_4,
#                             lora_5, strength_5, lora_6, strength_6, lora_7, strength_7, lora_8, strength_8, save_metadata=True, convert_to="diffusion_model", max_dimension=64):
#         print("Save_folder: {}".format(save_folder))
#         os.makedirs(save_folder, exist_ok = True)
#
#         temp_loras = {}
#         metadata = {"loras": "{} and {} and {} and {} and {} and {} and {} and {}".format(lora_1, lora_2, lora_3, lora_4, lora_5, lora_6, lora_7, lora_8)}
#         metadata["strengths"] = "{} and {} and {} and {} and {} and {} and {} and {}".format(strength_1, strength_2, strength_3, strength_4, strength_5, strength_6, strength_7, strength_8)
#         metadata["seed"] = "{}".format(seed)
#         metadata["num_output"] = "{}".format(num_output)
#
#         if (lora_1 != "None") and (strength_1 > 0.0):
#             print(lora_1)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_1, 1.0, "all")
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             temp_loras["1"] = {"lora_weights": lora_weights, "strength": strength_1}
#
#             for key in lora_weights.keys():
#                 print("LORA 1: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                 break
#
#             lora_weights = None
#
#         if (lora_2 != "None") and (strength_2 > 0.0):
#             print(lora_2)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_2, 1.0, "all")
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             temp_loras["2"] = {"lora_weights": lora_weights, "strength": strength_2}
#
#             for key in lora_weights.keys():
#                 print("LORA 2: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                 break
#
#             lora_weights = None
#
#         if (lora_3 != "None") and (strength_3 > 0.0):
#             print(lora_3)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_3, 1.0, "all")
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             temp_loras["3"] = {"lora_weights": lora_weights, "strength": strength_3}
#
#             for key in lora_weights.keys():
#                 print("LORA 3: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                 break
#
#             lora_weights = None
#
#         if (lora_4 != "None") and (strength_4 > 0.0):
#             print(lora_4)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_4, 1.0, "all")
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             temp_loras["4"] = {"lora_weights": lora_weights, "strength": strength_4}
#
#             for key in lora_weights.keys():
#                 print("LORA 4: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                 break
#
#             lora_weights = None
#
#         if (lora_5 != "None") and (strength_5 > 0.0):
#             print(lora_5)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_5, 1.0, "all")
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             temp_loras["5"] = {"lora_weights": lora_weights, "strength": strength_5}
#
#             for key in lora_weights.keys():
#                 print("LORA 5: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                 break
#
#             lora_weights = None
#
#         if (lora_6 != "None") and (strength_6 > 0.0):
#             print(lora_6)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_6, 1.0, "all")
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             temp_loras["6"] = {"lora_weights": lora_weights, "strength": strength_6}
#
#             for key in lora_weights.keys():
#                 print("LORA 6: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                 break
#
#             lora_weights = None
#
#         if (lora_7 != "None") and (strength_7 > 0.0):
#             print(lora_7)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_7, 1.0, "all")
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             temp_loras["7"] = {"lora_weights": lora_weights, "strength": strength_7}
#
#             for key in lora_weights.keys():
#                 print("LORA 7: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                 break
#
#             lora_weights = None
#
#         if (lora_8 != "None") and (strength_8 > 0.0):
#             print(lora_8)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_8, 1.0, "all")
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             temp_loras["8"] = {"lora_weights": lora_weights, "strength": strength_8}
#
#             for key in lora_weights.keys():
#                 print("LORA 8: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                 break
#
#             lora_weights = None
#
#         loras = {}
#
#         for lora_key in temp_loras.keys():
#             loras[lora_key] = {"lora_weights": {}, "strength": temp_loras[lora_key]["strength"]}
#
#             for key in temp_loras[lora_key]["lora_weights"].keys():
#                 new_key = key.replace("transformer.", "diffusion_model.")
#                 loras[lora_key]["lora_weights"][new_key] = temp_loras[lora_key]["lora_weights"][key].to(dtype=torch.bfloat16)
#
#         block_types = self.determine_lora_block_types(loras)
#         merge_mixtures, block_metadata = self.get_mixtures(seed, num_output, loras.keys(), block_types)
#
#         metadata["max_dimension"] = "{}".format(max_dimension)
#         metadata["block_types"] = "{}".format(block_types)
#
#         print("Max Dimension: {}".format(max_dimension))
#
#         save_message = ""
#
#         for mixture_key in merge_mixtures:
#             new_lora = {}
#             output_filename = os.path.join(save_folder, "{}_{:05}.safetensors".format(model_prefix, int(mixture_key)))
#
#             metadata["merge_mixture"] = "{}".format(merge_mixtures[mixture_key])
#             # metadata["block_metadata"] = "{}".format(block_metadata[int(mixture_key)])
#
#             for lora_key in loras.keys():
#                 mixture_single_blocks = merge_mixtures[mixture_key][lora_key]["single"]
#                 mixture_double_blocks = merge_mixtures[mixture_key][lora_key]["double"]
#
#                 for key in loras[lora_key]["lora_weights"].keys():
#                     temp_strings = str(key).split('.')
#                     temp_block_num = int(temp_strings[2])
#
#                     if temp_strings[1] == "single_blocks":
#                         if temp_block_num in mixture_single_blocks:
#                             new_lora[key] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"]).to(dtype=torch.bfloat16)
#                         continue
#
#                     if temp_strings[1] == "double_blocks":
#                         if temp_block_num in mixture_double_blocks:
#                             new_lora[key] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"]).to(dtype=torch.bfloat16)
#
#             if not save_metadata:
#                 metadata = None
#
#             print("Saving Model To: {}...".format(output_filename))
#             utils.save_torch_file(new_lora, output_filename, metadata=metadata)
#             print("Saving Model To: {}...Done.".format(output_filename))
#
#             save_message = "{}\n{}".format(save_message, "Weights Saved To: {}".format(output_filename))
#
#             new_lora = None
#             mm.soft_empty_cache()
#             gc.collect()
#             time.sleep(1)
#
#         return {"ui": {"tags": ["save_message"]}}

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
                "strength_1": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "blocks_1": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
                "lora_2": (['None'] + get_lora_list(),),
                "strength_2": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "blocks_2": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
                "lora_3": (['None'] + get_lora_list(),),
                "strength_3": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "blocks_3": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
                "lora_4": (['None'] + get_lora_list(),),
                "strength_4": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "blocks_4": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
                "lora_5": (['None'] + get_lora_list(),),
                "strength_5": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "blocks_5": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
                "lora_6": (['None'] + get_lora_list(),),
                "strength_6": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "blocks_6": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
                "lora_7": (['None'] + get_lora_list(),),
                "strength_7": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "blocks_7": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
                "lora_8": (['None'] + get_lora_list(),),
                "strength_8": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                "blocks_8": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
                "save_metadata": ("BOOLEAN", {"default": True}),
                "convert_to": (["diffusion_model", "transformer"], {"default": "diffusion_model"}),
                "max_dimension": ([32, 64, 128], {"default": 128}),
                "discard_linear": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "merge_multiple_loras"
    CATEGORY = "Warped/Hunyuan/Mixers"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

    def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load and filter a single LoRA model."""
        if not lora_name or strength == 0:
            return {}, {}

        # # Load the LoRA weights
        lora_weights = warped_load_lora_weights(lora_name)
        lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")

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

        at_least_one = False

        while not at_least_one:
            if len(double_block_loras) > 0:
                for key in lora_keys:
                    lowest = 999
                    highest = -1
                    lowest_mixture_key = None
                    highest_mixture_key = None

                    for mixture_key in mixtures:
                        if len(mixtures[mixture_key][key]["double"]) < lowest:
                            lowest = len(mixtures[mixture_key][key]["double"])
                            lowest_mixture_key = mixture_key

                        if len(mixtures[mixture_key][key]["double"]) > highest:
                            highest = len(mixtures[mixture_key][key]["double"])
                            highest_mixture_key = mixture_key

                    if lowest < 1:
                        print("low key: {}".format(lowest_mixture_key))
                        print("high key: {}".format(highest_mixture_key))
                        mixtures[lowest_mixture_key][key]["double"].append(mixtures[highest_mixture_key][key]["double"][0])
                        print("low mix: {}".format(mixtures[lowest_mixture_key][key]["double"]))
                        print("high mixes before: {}".format(mixtures[highest_mixture_key][key]["double"]))
                        mixtures[highest_mixture_key][key]["double"].pop(0)
                        print("high mixes after: {}".format(mixtures[highest_mixture_key][key]["double"]))
                    else:
                        at_least_one = True
            else:
                at_least_one = True

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

    def merge_multiple_loras(self, save_folder, model_prefix, seed, num_output, lora_1, strength_1, blocks_1, lora_2, strength_2, blocks_2, lora_3, strength_3, blocks_3, lora_4, strength_4, blocks_4,
                            lora_5, strength_5, blocks_5, lora_6, strength_6, blocks_6, lora_7, strength_7, blocks_7, lora_8, strength_8, blocks_8, save_metadata=True, convert_to="diffusion_model",
                            max_dimension=128, discard_linear=True):
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

            if (blocks_1 == "single_blocks") or (blocks_1 == "double_blocks"):
                lora_weights = filter_lora_keys(lora_weights, blocks_1)

            lora_weights = convert_lora_dimensions(max_dimension, lora_weights)

            print_it = True
            for key in lora_weights.keys():
                if print_it:
                    print("LORA 1: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
                    print_it = False

                lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)

            temp_loras["1"] = {"lora_weights": lora_weights, "strength": strength_1}

            lora_weights = None

        if (lora_2 != "None") and (strength_2 > 0.0):
            print(lora_2)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_2, 1.0, "all")

            if (blocks_2 == "single_blocks") or (blocks_2 == "double_blocks"):
                lora_weights = filter_lora_keys(lora_weights, blocks_2)

            lora_weights = convert_lora_dimensions(max_dimension, lora_weights)

            print_it = True
            for key in lora_weights.keys():
                if print_it:
                    print("LORA 2: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
                    print_it = False

                lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)

            temp_loras["2"] = {"lora_weights": lora_weights, "strength": strength_2}

            lora_weights = None

        if (lora_3 != "None") and (strength_3 > 0.0):
            print(lora_3)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_3, 1.0, "all")

            if (blocks_3 == "single_blocks") or (blocks_3 == "double_blocks"):
                lora_weights = filter_lora_keys(lora_weights, blocks_3)

            lora_weights = convert_lora_dimensions(max_dimension, lora_weights)

            print_it = True
            for key in lora_weights.keys():
                if print_it:
                    print("LORA 3: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
                    print_it = False

                lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)

            temp_loras["3"] = {"lora_weights": lora_weights, "strength": strength_3}

            lora_weights = None

        if (lora_4 != "None") and (strength_4 > 0.0):
            print(lora_4)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_4, 1.0, "all")

            if (blocks_4 == "single_blocks") or (blocks_4 == "double_blocks"):
                lora_weights = filter_lora_keys(lora_weights, blocks_4)

            lora_weights = convert_lora_dimensions(max_dimension, lora_weights)

            print_it = True
            for key in lora_weights.keys():
                if print_it:
                    print("LORA 4: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
                    print_it = False

                lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)

            temp_loras["4"] = {"lora_weights": lora_weights, "strength": strength_4}

            lora_weights = None

        if (lora_5 != "None") and (strength_5 > 0.0):
            print(lora_5)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_5, 1.0, "all")

            if (blocks_5 == "single_blocks") or (blocks_5 == "double_blocks"):
                lora_weights = filter_lora_keys(lora_weights, blocks_5)

            lora_weights = convert_lora_dimensions(max_dimension, lora_weights)

            print_it = True
            for key in lora_weights.keys():
                if print_it:
                    print("LORA 5: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
                    print_it = False

                lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)

            temp_loras["5"] = {"lora_weights": lora_weights, "strength": strength_5}

            lora_weights = None

        if (lora_6 != "None") and (strength_6 > 0.0):
            print(lora_6)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_6, 1.0, "all")

            if (blocks_6 == "single_blocks") or (blocks_6 == "double_blocks"):
                lora_weights = filter_lora_keys(lora_weights, blocks_6)

            lora_weights = convert_lora_dimensions(max_dimension, lora_weights)

            print_it = True
            for key in lora_weights.keys():
                if print_it:
                    print("LORA 6: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
                    print_it = False

                lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)

            temp_loras["6"] = {"lora_weights": lora_weights, "strength": strength_6}

            lora_weights = None

        if (lora_7 != "None") and (strength_7 > 0.0):
            print(lora_7)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_7, 1.0, "all")

            if (blocks_7 == "single_blocks") or (blocks_7 == "double_blocks"):
                lora_weights = filter_lora_keys(lora_weights, blocks_7)

            lora_weights = convert_lora_dimensions(max_dimension, lora_weights)

            print_it = True
            for key in lora_weights.keys():
                if print_it:
                    print("LORA 7: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
                    print_it = False

                lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)

            temp_loras["7"] = {"lora_weights": lora_weights, "strength": strength_7}

            lora_weights = None

        if (lora_8 != "None") and (strength_8 > 0.0):
            print(lora_8)
            # Load and filter the LoRA weights
            lora_weights = self.load_lora(lora_8, 1.0, "all")

            if (blocks_8 == "single_blocks") or (blocks_8 == "double_blocks"):
                lora_weights = filter_lora_keys(lora_weights, blocks_8)

            lora_weights = convert_lora_dimensions(max_dimension, lora_weights)

            print_it = True
            for key in lora_weights.keys():
                if print_it:
                    print("LORA 8: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
                    print_it = False

                lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)

            temp_loras["8"] = {"lora_weights": lora_weights, "strength": strength_8}

            lora_weights = None

        loras = {}

        for lora_key in temp_loras.keys():
            loras[lora_key] = {"lora_weights": {}, "strength": temp_loras[lora_key]["strength"]}

            for key in temp_loras[lora_key]["lora_weights"].keys():
                new_key = key.replace("transformer.", "diffusion_model.")
                loras[lora_key]["lora_weights"][new_key] = temp_loras[lora_key]["lora_weights"][key].to(dtype=torch.bfloat16)

        block_types = self.determine_lora_block_types(loras)
        merge_mixtures, block_metadata = self.get_mixtures(seed, num_output, loras.keys(), block_types)

        metadata["max_dimension"] = "{}".format(max_dimension)
        metadata["block_types"] = "{}".format(block_types)

        print("Max Dimension: {}".format(max_dimension))

        save_message = ""

        for mixture_key in merge_mixtures:
            time.sleep(1)
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
                            new_lora[key] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"]).to(dtype=torch.bfloat16)
                        continue

                    if temp_strings[1] == "double_blocks":
                        if temp_block_num in mixture_double_blocks:
                            new_lora[key] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"]).to(dtype=torch.bfloat16)

            if not save_metadata:
                metadata = None

            if discard_linear:
                temp_lora = new_lora.copy()

                for temp_key in temp_lora:
                    if "linear" in temp_key:
                        del new_lora[temp_key]

            print("Saving Model To: {}...".format(output_filename))
            utils.save_torch_file(new_lora, output_filename, metadata=metadata)
            print("Saving Model To: {}...Done.".format(output_filename))

            save_message = "{}\n{}".format(save_message, "Weights Saved To: {}".format(output_filename))

            new_lora = None
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

        return {"ui": {"tags": ["save_message"]}}

# class WarpedHunyuanMultiLoraMixerPlatinum:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "save_folder": ("STRING", {"default": get_default_output_folder()}),
#                 "model_prefix": ("STRING", {"default": "new_model_hy"}),
#                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
#                 "num_output": ("INT", {"default": 1, "min": 1, "max": 100}),
#                 "lora_1": (['None'] + get_lora_list(),),
#                 "strength_1": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "blocks_1": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
#                 "lora_2": (['None'] + get_lora_list(),),
#                 "strength_2": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "blocks_2": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
#                 "lora_3": (['None'] + get_lora_list(),),
#                 "strength_3": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "blocks_3": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
#                 "lora_4": (['None'] + get_lora_list(),),
#                 "strength_4": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "blocks_4": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
#                 "lora_5": (['None'] + get_lora_list(),),
#                 "strength_5": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "blocks_5": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
#                 "lora_6": (['None'] + get_lora_list(),),
#                 "strength_6": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "blocks_6": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
#                 "lora_7": (['None'] + get_lora_list(),),
#                 "strength_7": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "blocks_7": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
#                 "lora_8": (['None'] + get_lora_list(),),
#                 "strength_8": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
#                 "blocks_8": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
#                 "save_metadata": ("BOOLEAN", {"default": True}),
#                 "convert_to": (["diffusion_model", "transformer"], {"default": "diffusion_model"}),
#                 "max_dimension": ([32, 64, 128], {"default": 128}),
#                 "remove_linear": ("BOOLEAN", {"default": True}),
#                 "remove_single_blocks": ("BOOLEAN", {"default": True}),
#             },
#         }
#
#     RETURN_TYPES = ()
#     OUTPUT_NODE = True
#     OUTPUT_IS_LIST = (True,)
#     FUNCTION = "merge_multiple_loras"
#     CATEGORY = "Warped/Hunyuan/Mixers"
#     DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."
#
#     def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
#         """Load and filter a single LoRA model."""
#         if not lora_name or strength == 0:
#             return {}, {}
#
#         # Get the full path to the LoRA file
#         lora_path = folder_paths.get_full_path("loras", lora_name)
#         if not os.path.exists(lora_path):
#             raise ValueError(f"LoRA file not found: {lora_path}")
#
#         # Load the LoRA weights
#         lora_weights = utils.load_torch_file(lora_path)
#
#         lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
#
#         return lora_weights
#
#     def get_random_key(self, keys):
#         if len(keys) > 0:
#             random_key = random.randint(0, len(keys) - 1)
#             # print("Keys: {}  |  Random Key: {}".format(random_key))
#             return keys[random_key]
#
#         return -1
#
#     def get_mixtures(self, seed, num_output, lora_keys, block_types):
#         random.seed(seed)
#         mixtures = {}
#
#         for i in range(num_output):
#             mixtures["{}".format(i + 1)] = {}
#
#         single_block_loras = []
#         double_block_loras = []
#
#         for key in lora_keys:
#             for mixture_key in mixtures.keys():
#                 mixtures[mixture_key][key] = {"single": [], "double": []}
#
#                 if block_types[key]["has_single_blocks"]:
#                     single_block_loras.append(int(key))
#
#                 if block_types[key]["has_double_blocks"]:
#                     double_block_loras.append(int(key))
#
#         for mixture_key in mixtures.keys():
#             if len(single_block_loras) > 0:
#                 for j in range(40):
#                     random_key = self.get_random_key(single_block_loras)
#                     temp_key = "{}".format(random_key)
#                     mixtures[mixture_key][temp_key]["single"].append(j)
#
#             if len(double_block_loras) > 0:
#                 for j in range(20):
#                     random_key = self.get_random_key(double_block_loras)
#                     temp_key = "{}".format(random_key)
#                     mixtures[mixture_key][temp_key]["double"].append(j)
#
#             i += 1
#
#         at_least_one = False
#
#         while not at_least_one:
#             if len(double_block_loras) > 0:
#                 for key in lora_keys:
#                     lowest = 999
#                     highest = -1
#                     lowest_mixture_key = None
#                     highest_mixture_key = None
#
#                     for mixture_key in mixtures:
#                         if len(mixtures[mixture_key][key]["double"]) < lowest:
#                             lowest = len(mixtures[mixture_key][key]["double"])
#                             lowest_mixture_key = mixture_key
#
#                         if len(mixtures[mixture_key][key]["double"]) > highest:
#                             highest = len(mixtures[mixture_key][key]["double"])
#                             highest_mixture_key = mixture_key
#
#                     if lowest < 1:
#                         print("low key: {}".format(lowest_mixture_key))
#                         print("high key: {}".format(highest_mixture_key))
#                         mixtures[lowest_mixture_key][key]["double"].append(mixtures[highest_mixture_key][key]["double"][0])
#                         print("low mix: {}".format(mixtures[lowest_mixture_key][key]["double"]))
#                         print("high mixes before: {}".format(mixtures[highest_mixture_key][key]["double"]))
#                         mixtures[highest_mixture_key][key]["double"].pop(0)
#                         print("high mixes after: {}".format(mixtures[highest_mixture_key][key]["double"]))
#                     else:
#                         at_least_one = True
#             else:
#                 at_least_one = True
#
#         print("\nMixtures\n")
#
#         block_metadata = ""
#
#         for mixture_key in mixtures.keys():
#             for key in mixtures[mixture_key]:
#                 print("{} | {}: {}".format(mixture_key, key, mixtures[mixture_key][key]))
#
#                 if len(block_metadata) > 0:
#                     block_metadata = "{}  |  {}".format(block_metadata, "{}: {}: {}".format(mixture_key, key, mixtures[mixture_key][key]))
#                 else:
#                     block_metadata = "{}: {}: {}".format(mixture_key, key, mixtures[mixture_key][key])
#
#             print("\n")
#
#         return mixtures, block_metadata
#
#     def determine_lora_block_types(self, loras):
#         block_types = {}
#
#         for lora_key in loras.keys():
#             block_types[lora_key] = { "has_single_blocks": False, "has_double_blocks": False }
#
#             for key in loras[lora_key]["lora_weights"].keys():
#                 if "single_blocks" in key:
#                     block_types[lora_key]["has_single_blocks"] = True
#                 elif "double_blocks" in key:
#                     block_types[lora_key]["has_double_blocks"] = True
#
#                 if block_types[lora_key]["has_single_blocks"] and block_types[lora_key]["has_double_blocks"]:
#                     break
#
#         return block_types
#
#     def merge_multiple_loras(self, save_folder, model_prefix, seed, num_output, lora_1, strength_1, blocks_1, lora_2, strength_2, blocks_2, lora_3, strength_3, blocks_3, lora_4, strength_4, blocks_4,
#                             lora_5, strength_5, blocks_5, lora_6, strength_6, blocks_6, lora_7, strength_7, blocks_7, lora_8, strength_8, blocks_8, save_metadata=True, convert_to="diffusion_model",
#                             max_dimension=64, remove_linear=True, remove_single_blocks=True):
#         print("Save_folder: {}".format(save_folder))
#         os.makedirs(save_folder, exist_ok = True)
#
#         temp_loras = {}
#         metadata = {"loras": "{} and {} and {} and {} and {} and {} and {} and {}".format(lora_1, lora_2, lora_3, lora_4, lora_5, lora_6, lora_7, lora_8)}
#         metadata["strengths"] = "{} and {} and {} and {} and {} and {} and {} and {}".format(strength_1, strength_2, strength_3, strength_4, strength_5, strength_6, strength_7, strength_8)
#         metadata["seed"] = "{}".format(seed)
#         metadata["num_output"] = "{}".format(num_output)
#
#         if (lora_1 != "None") and (strength_1 > 0.0):
#             print(lora_1)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_1, 1.0, "all")
#
#             if (blocks_1 == "single_blocks") or (blocks_1 == "double_blocks"):
#                 lora_weights = filter_lora_keys(lora_weights, blocks_1)
#
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             print_it = True
#             for key in lora_weights.keys():
#                 if print_it:
#                     print("LORA 1: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                     print_it = False
#
#                 lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)
#
#             temp_loras["1"] = {"lora_weights": lora_weights, "strength": strength_1}
#
#             lora_weights = None
#
#         if (lora_2 != "None") and (strength_2 > 0.0):
#             print(lora_2)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_2, 1.0, "all")
#
#             if (blocks_2 == "single_blocks") or (blocks_2 == "double_blocks"):
#                 lora_weights = filter_lora_keys(lora_weights, blocks_2)
#
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             print_it = True
#             for key in lora_weights.keys():
#                 if print_it:
#                     print("LORA 2: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                     print_it = False
#
#                 lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)
#
#             temp_loras["2"] = {"lora_weights": lora_weights, "strength": strength_2}
#
#             lora_weights = None
#
#         if (lora_3 != "None") and (strength_3 > 0.0):
#             print(lora_3)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_3, 1.0, "all")
#
#             if (blocks_3 == "single_blocks") or (blocks_3 == "double_blocks"):
#                 lora_weights = filter_lora_keys(lora_weights, blocks_3)
#
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             print_it = True
#             for key in lora_weights.keys():
#                 if print_it:
#                     print("LORA 3: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                     print_it = False
#
#                 lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)
#
#             temp_loras["3"] = {"lora_weights": lora_weights, "strength": strength_3}
#
#             lora_weights = None
#
#         if (lora_4 != "None") and (strength_4 > 0.0):
#             print(lora_4)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_4, 1.0, "all")
#
#             if (blocks_4 == "single_blocks") or (blocks_4 == "double_blocks"):
#                 lora_weights = filter_lora_keys(lora_weights, blocks_4)
#
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             print_it = True
#             for key in lora_weights.keys():
#                 if print_it:
#                     print("LORA 4: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                     print_it = False
#
#                 lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)
#
#             temp_loras["4"] = {"lora_weights": lora_weights, "strength": strength_4}
#
#             lora_weights = None
#
#         if (lora_5 != "None") and (strength_5 > 0.0):
#             print(lora_5)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_5, 1.0, "all")
#
#             if (blocks_5 == "single_blocks") or (blocks_5 == "double_blocks"):
#                 lora_weights = filter_lora_keys(lora_weights, blocks_5)
#
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             print_it = True
#             for key in lora_weights.keys():
#                 if print_it:
#                     print("LORA 5: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                     print_it = False
#
#                 lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)
#
#             temp_loras["5"] = {"lora_weights": lora_weights, "strength": strength_5}
#
#             lora_weights = None
#
#         if (lora_6 != "None") and (strength_6 > 0.0):
#             print(lora_6)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_6, 1.0, "all")
#
#             if (blocks_6 == "single_blocks") or (blocks_6 == "double_blocks"):
#                 lora_weights = filter_lora_keys(lora_weights, blocks_6)
#
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             print_it = True
#             for key in lora_weights.keys():
#                 if print_it:
#                     print("LORA 6: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                     print_it = False
#
#                 lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)
#
#             temp_loras["6"] = {"lora_weights": lora_weights, "strength": strength_6}
#
#             lora_weights = None
#
#         if (lora_7 != "None") and (strength_7 > 0.0):
#             print(lora_7)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_7, 1.0, "all")
#
#             if (blocks_7 == "single_blocks") or (blocks_7 == "double_blocks"):
#                 lora_weights = filter_lora_keys(lora_weights, blocks_7)
#
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             print_it = True
#             for key in lora_weights.keys():
#                 if print_it:
#                     print("LORA 7: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                     print_it = False
#
#                 lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)
#
#             temp_loras["7"] = {"lora_weights": lora_weights, "strength": strength_7}
#
#             lora_weights = None
#
#         if (lora_8 != "None") and (strength_8 > 0.0):
#             print(lora_8)
#             # Load and filter the LoRA weights
#             lora_weights = self.load_lora(lora_8, 1.0, "all")
#
#             if (blocks_8 == "single_blocks") or (blocks_8 == "double_blocks"):
#                 lora_weights = filter_lora_keys(lora_weights, blocks_8)
#
#             lora_weights = convert_lora_dimensions(max_dimension, lora_weights)
#
#             print_it = True
#             for key in lora_weights.keys():
#                 if print_it:
#                     print("LORA 8: {}  |  Sample Shape: {}".format(key, lora_weights[key].shape))
#                     print_it = False
#
#                 lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)
#
#             temp_loras["8"] = {"lora_weights": lora_weights, "strength": strength_8}
#
#             lora_weights = None
#
#         loras = {}
#
#         for lora_key in temp_loras.keys():
#             loras[lora_key] = {"lora_weights": {}, "strength": temp_loras[lora_key]["strength"]}
#
#             for key in temp_loras[lora_key]["lora_weights"].keys():
#                 new_key = key.replace("transformer.", "diffusion_model.")
#                 loras[lora_key]["lora_weights"][new_key] = temp_loras[lora_key]["lora_weights"][key].to(dtype=torch.bfloat16)
#
#         block_types = self.determine_lora_block_types(loras)
#         merge_mixtures, block_metadata = self.get_mixtures(seed, num_output, loras.keys(), block_types)
#
#         metadata["max_dimension"] = "{}".format(max_dimension)
#         metadata["block_types"] = "{}".format(block_types)
#
#         print("Max Dimension: {}".format(max_dimension))
#
#         save_message = ""
#
#         for mixture_key in merge_mixtures:
#             time.sleep(1)
#             new_lora = {}
#             output_filename = os.path.join(save_folder, "{}_{:05}.safetensors".format(model_prefix, int(mixture_key)))
#
#             metadata["merge_mixture"] = "{}".format(merge_mixtures[mixture_key])
#             # metadata["block_metadata"] = "{}".format(block_metadata[int(mixture_key)])
#
#             for lora_key in loras.keys():
#                 mixture_single_blocks = merge_mixtures[mixture_key][lora_key]["single"]
#                 mixture_double_blocks = merge_mixtures[mixture_key][lora_key]["double"]
#
#                 for key in loras[lora_key]["lora_weights"].keys():
#                     temp_strings = str(key).split('.')
#                     temp_block_num = int(temp_strings[2])
#
#                     if temp_strings[1] == "single_blocks":
#                         if temp_block_num in mixture_single_blocks:
#                             new_lora[key] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"]).to(dtype=torch.bfloat16)
#                         continue
#
#                     if temp_strings[1] == "double_blocks":
#                         if temp_block_num in mixture_double_blocks:
#                             new_lora[key] = torch.mul(loras[lora_key]["lora_weights"][key], loras[lora_key]["strength"]).to(dtype=torch.bfloat16)
#
#             dummy_lora = new_lora,copy()
#
#             for key in dummy_lora:
#                 if "single_blocks" in key:
#                     if remove_single_blocks:
#                         del new_lora[key]
#                     elif remove_linear:
#                         if "linear" in key:
#                             del new_lora[key]
#
#                     continue
#
#                 if remove_linear:
#                     if "linear" in key:
#                         del new_lora[key]
#
#             if not save_metadata:
#                 metadata = None
#
#             print("Saving Model To: {}...".format(output_filename))
#             utils.save_torch_file(new_lora, output_filename, metadata=metadata)
#             print("Saving Model To: {}...Done.".format(output_filename))
#
#             save_message = "{}\n{}".format(save_message, "Weights Saved To: {}".format(output_filename))
#
#             new_lora = None
#             mm.soft_empty_cache()
#             gc.collect()
#             time.sleep(1)
#
#         return {"ui": {"tags": ["save_message"]}}

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
    CATEGORY = "Warped/Wan/Merge"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

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

    CATEGORY = "Warped/General/Image"

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

    CATEGORY = "Warped/General/Image"

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

    CATEGORY = "Warped/General/Image"

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

    CATEGORY = "Warped/General/Image"

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

    CATEGORY = "Warped/General/Latent"

    def generate(self, batch_size, width, height):
        temp_latent = torch.zeros([1, 16, int(((batch_size - 1) / 4) + 1), int(height // 8), int(width // 8)], dtype=torch.float32, device=self.offload_device)

        print("Empty Latent Batch Shape: {}".format(temp_latent.shape))

        if len(temp_latent.shape) < 5:
            temp_latent = temp_latent.unsqueeze(0)

        return ({"samples": temp_latent}, batch_size, )

class WarpedHunyuanVideoToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"video_batch": ("IMAGE", ),
                             "positive": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "guidance_type": (["v1 (concat)", "v2 (replace)", "custom"], )
                            },
                }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"

    CATEGORY = "Warped/Hunyuan/Conditioning"

    def encode(self, video_batch, positive, vae, guidance_type):
        mm.unload_all_models()

        out_latent = {}

        batch_size = 1
        width = video_batch.shape[2]
        height = video_batch.shape[1]
        length = video_batch.shape[0]

        print("width: {}  |  height: {}  |  length: {}".format(width, height, length))

        video_batch = comfy.utils.common_upscale(video_batch[:length, :, :, :3].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        latent = partial_encode_tiled(vae, video_batch) #vae.encode(video_batch)

        print("latents Shape: {}".format(latent.shape))

        video_tuple = torch.split(video_batch, 1, dim=0)
        video_split = [item for item in video_tuple]

        start_image = video_split[0]

        video_tuple = None
        video_split = None

        concat_latent_image = vae.encode(start_image)
        mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
        mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

        if guidance_type == "v1 (concat)":
            cond = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        elif guidance_type == "v2 (replace)":
            cond = {'guiding_frame_index': 0}
            latent[:, :, :concat_latent_image.shape[2]] = concat_latent_image
            out_latent["noise_mask"] = mask
        elif guidance_type == "custom":
            cond = {"ref_latent": concat_latent_image}

        positive = node_helpers.conditioning_set_values(positive, cond)

        out_latent["samples"] = latent
        return (positive, out_latent)

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

    RETURN_TYPES = ("IMAGE", "LATENT", "INT", "STRING", "BOOLEAN", )
    RETURN_NAMES = ("images", "latents", "seed", "generation_status", "valid_output", )

    FUNCTION = "sample"

    CATEGORY = "Warped/General/Sampling"

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
        valid_output = False

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
            valid_output = True
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
            print("WarpedSamplerCustomAdv: Returning only partial results (if any).\n If zero images generated, a blank yellow image will be returned, and valid_output will indicate False.\n")
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

            generation_status = f"\nWarpedSamplerCustomAdv: Processing Interrupted."

            traceback.print_tb(ie.__traceback__, limit=99, file=sys.stdout)

            raise mm.InterruptProcessingException(f"WarpedSamplerCustomAdv: Processing Interrupted.")

            pass

        except BaseException as e:
            print(f"\nWarpedSamplerCustomAdv: Exception During Processing: {str(e)}")
            print("WarpedSamplerCustomAdv: Returning only partial results (if any).\n If zero images generated, a blank red image will be returned, and valid_output will indicate False.\n")
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

            generation_status = f"WarpedSamplerCustomAdv: Exception During Processing: {str(e)}"
            generation_status = "{}{}".format(generation_status, "WarpedSamplerCustomAdv: Returning only partial results (if any).\nIf zero images generated, a blank red image will be returned, and valid_output will indicate False.")

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

        return (output_images, {"samples": output_images_latents}, self.seed, generation_status, valid_output, )

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
                    "is_v2v": ("BOOLEAN", {"default": False}),
                    "output_latents": ("BOOLEAN", {"default": False}),
                    }
                }

    RETURN_TYPES = ("IMAGE", "LATENT", "INT", "STRING", "BOOLEAN", )
    RETURN_NAMES = ("images", "latents", "seed", "generation_status", "valid_output", )

    FUNCTION = "sample"

    CATEGORY = "Warped/General/Sampling"

    def sample(self, latent, vae, seed, guider, sampler, sigmas, dec_tile_size, dec_overlap, dec_temporal_size, dec_temporal_overlap,
                    skip_frames, noise_scale, scaling_strength=1.0, is_v2v=False, output_latents=False):
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
        self.init_mode = "image"

        if is_v2v:
            self.init_mode = "video"

        callback = self.setup_callbacks()
        disable_pbar = not utils.PROGRESS_BAR_ENABLED

        print("\nSigmas: {}".format(self.sigmas))

        latents = latent["samples"]

        if len(latents.shape) < 5:
            latents = latents.unsqueeze(0)

        num_frames = int(((latents.shape[2] - 1) * 4) + 1)
        self.latents_depth = latents.shape[1]

        print("latents_depth: {}".format(self.latents_depth))

        self.width = latents.shape[4]
        self.height = latents.shape[3]
        print("\nDecoded Width is {}  |  Decoded Height is {}".format(int(self.width * 8), int(self.height * 8)))

        generation_status = ""

        noise_latents = self.initialize_frames(latents, init_mode=self.init_mode)

        print("-------------------------------------------------------------------------------------------")
        print("WarpedSamplerCustomAdvLatent: Latents Shape: {}  |  Noise Latents Shape: {}".format(latents.shape, noise_latents.shape))
        print("-------------------------------------------------------------------------------------------")

        output_images = None
        output_images_latents = None
        interrupted = False
        valid_output = False

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
            valid_output = True

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
            print("WarpedSamplerCustomAdvLatent: Returning only partial results (if any).\n If zero images generated, a blank yellow image will be returned, and valid_output will indicate False.\n")
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

            generation_status = f"\nWarpedSamplerCustomAdvLatent: Processing Interrupted."

            traceback.print_tb(ie.__traceback__, limit=99, file=sys.stdout)

            raise mm.InterruptProcessingException(f"WarpedSamplerCustomAdvLatent: Processing Interrupted.")
            pass

        except BaseException as e:
            print(f"\nWarpedSamplerCustomAdvLatent: Exception During Processing: {str(e)}")
            print("WarpedSamplerCustomAdvLatent: Returning only partial results (if any).\n If zero images generated, a blank red image will be returned, and valid_output will indicate False.\n")
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

            generation_status = f"WarpedSamplerCustomAdvLatent: Exception During Processing: {str(e)}"
            generation_status = "{}{}".format(generation_status, "WarpedSamplerCustomAdvLatent: Returning only partial results (if any).\nIf zero images generated, a blank red image will be returned, and valid_output will indicate False.")

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
            output_images_latents = torch.zeros([1, self.latents_depth, 1, self.height, self.width], dtype=torch.float32, device=self.offload_device)

        return (output_images, {"samples": output_images_latents}, self.seed, generation_status, valid_output, )

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
        pad_frames = torch.zeros([1, self.latents_depth, num_frames, self.height, self.width], dtype=torch.float32, device=self.offload_device)
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
        noise_latents_full = torch.zeros([1, self.latents_depth, int(frame_count), self.height, self.width], dtype=torch.float32, device=self.offload_device)

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

    def initialize_frames(self, latents, init_mode="image"):
        if len(latents.shape) < 5:
            latents = latents.unsqueeze(0)

        print("WarpedSamplerCustomAdvLatent: Encoded latents_full Shape: {}".format(latents.shape))
        latents_full = latents.clone().detach()
        latents_full = latents_full.to(dtype=torch.float32, device=self.offload_device)

        if init_mode == "image":
            noise_latents_full = self.initialize_noise(latents_full.shape[2])
        else:
            if Decimal(self.noise_scale).compare(Decimal(0.00)) != 0:
                noise_latents_full = warped_prepare_noise(latents_full, self.seed)
                print("WarpedSamplerCustomAdvLatent: noise_latents_full Shape: {}".format(noise_latents_full.shape))

                noise_latents_full = torch.mul(noise_latents_full, self.noise_scale)

            if len(noise_latents_full.shape) < 5:
                noise_latents_full.unsqueeze(0)

            noise_latents_full = noise_latents_full.to(dtype=torch.float32, device=self.offload_device)

        mm.soft_empty_cache()
        gc.collect()
        time.sleep(0.1)

        return noise_latents_full

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

    CATEGORY = "Warped/Wan/Conditioning"

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

PROMPT_TEMPLATE_ENCODE_VIDEO_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

class WarpedSamplerCustomScripted:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "positive": ("STRING", {"default": None} ),
                    "negative": ("STRING", {"default": None}),
                    "clip": ("CLIP", ),
                    "preferred_frame_count": ("INT", {"default": 81, "min": 17, "max": nodes.MAX_RESOLUTION, "step": 4}),
                    "preferred_batch_size": ("INT", {"default": 17, "min": 17, "max": 301, "step": 4}),
                    "use_batch_size": (["next_lowest", "next_highest", "closest", "exact"], {"default": "next_lowest", "tooltip": "Number of frames generated may be impacted by choice."}),
                    "vae": ("VAE", ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "model": ("MODEL", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "dec_tile_size": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 32}),
                    "dec_overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                    "dec_temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to decode at a time."}),
                    "dec_temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap."}),
                    "skip_frames": ("INT", {"default": 0, "min": 0, "max": 32, "step": 4}),
                    "noise_scale": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                    "mode": (["Hunyuan", "Wan 2.1", "Wan 2.2 TI2V", "Wan 2.2 Standard"], {"default": "Hunyuan"}),
                    "hunyuan_guidance_type": (["v1 (concat)", "v2 (replace)", "custom"], {"default": "v2 (replace)", "tooltip": "Only used if mode is \"Hunyuan\"."}),
                    "hunyuan_image_interleave": ("INT", {"default": 1, "min": 1, "max": 512, "tooltip": "How much the image influences things vs the text prompt. Higher number means more influence from the text prompt."}),
                    },
                "optional":
                    {"noise_strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
                    "blend_frames": ("INT", {"default":0, "min":0, "max": 16, "step": 1}),
                    "t2v_width": ("INT", {"default":480, "min":256, "max": 1280, "step": 16}),
                    "t2v_height": ("INT", {"default":720, "min":256, "max": 1280, "step": 16}),
                    "scaling_strength": ("FLOAT", {"default": 1.0}),
                    "clip_vision_model": ("CLIP_VISION", ),
                    "start_image": ("IMAGE", ),
                    "flux_guidance": ("FLOAT", {"default": 7, "min": 0.0, "max": 100.0, "step": 0.1}),
                    "use_flux_guidance": ("BOOLEAN", {"default": False}),
                    "dummy_frames": ("INT", {"default":17, "min":17, "max": 301, "step": 4, "tooltip": "Number of frames to generate in dummy batch."}),
                    "gen_dummy": ("BOOLEAN", {"default": False, "tooltip": "For t2v or i2v only. Will generate a dummy batch to obtain a starting image for main generation."}),
                    "gen_dummy_only": ("BOOLEAN", {"default": False, "tooltip": "Will generate dummy batch only."}),
                    "use_dummy_image": (["first", "middle", "last", "random", "all"], {"default": "last", "tooltip": "Which dummy batch image to start main generation."}),
                    "output_latents": ("BOOLEAN", {"default": False}),
                    "verbose_messaging": ("BOOLEAN", {"default": False}),
                    "secondary_sigmas": ("SIGMAS", {"default": None}),
                    "secondary_model": ("MODEL", {"default": None}),
                    "batches_script": ("WARPEDSCRIPTS", {"default": None}),
                    }
                }

    RETURN_TYPES = ("IMAGE", "LATENT", "INT", "STRING", "IMAGE", "BOOLEAN",)
    RETURN_NAMES = ("images", "latents", "seed", "generation_status", "image_used", "valid_output",)

    FUNCTION = "sample"

    CATEGORY = "Warped/General/Sampling"

    def sample(self, positive, negative, clip, preferred_frame_count, preferred_batch_size, use_batch_size, vae, seed, model, sampler, sigmas, dec_tile_size, dec_overlap, dec_temporal_size, dec_temporal_overlap,
                    skip_frames, noise_scale, mode="Hunyuan", hunyuan_guidance_type="v2 (replace)", hunyuan_image_interleave=2, noise_strength=1.0, blend_frames=5, t2v_width=480, t2v_height=720, scaling_strength=1.0,
                    clip_vision_model=None, start_image=None, flux_guidance=3.5, dummy_frames=17, gen_dummy=False, gen_dummy_only=False, use_dummy_image="last", use_flux_guidance=False, output_latents=False, verbose_messaging=False,
                    secondary_sigmas=None, secondary_model=None, batches_script=None):

        batch_scripts = batches_script

        if (positive is None) and (batches_script is None):
            raise ValueError("positive and batches_script cannot both be None.")
        # elif not batches_script is None:
        #     if verbose_messaging:
        #         print("\n{}\n".format(batches_script))
        #
        #     batch_scripts = {}
        #     temp_strings = batches_script.split("||")
        #
        #     for entry in temp_strings:
        #         temp_script = entry.strip().split("|")
        #         batch_scripts[temp_script[0].strip()] = temp_script[1].strip()
        #
        #     if verbose_messaging:
        #         print("\n{}\n".format(batch_scripts))

        self.device = mm.get_torch_device()
        self.offload_device = get_offload_device()
        mm.unload_all_models()
        gc.collect()
        time.sleep(1)

        self.positive = positive
        self.negative = negative
        self.clip = clip
        self.use_clip = clip.clone()
        self.preferred_frame_count = preferred_frame_count
        self.preferred_batch_size = preferred_batch_size
        self.vae = vae
        self.seed = seed
        self.sampler = sampler
        self.sigmas = sigmas
        self.secondary_sigmas = secondary_sigmas
        self.noise_scale = noise_scale
        self.mode = mode
        self.dec_tile_size = dec_tile_size
        self.dec_overlap = dec_overlap
        self.dec_temporal_size = dec_temporal_size
        self.dec_temporal_overlap = dec_temporal_overlap
        self.clip_vision = clip_vision_model
        self.noise_strength = noise_strength
        self.blend_frames = blend_frames
        self.t2v_width = t2v_width
        self.t2v_height = t2v_height
        self.dummy_frames = dummy_frames
        self.dummy_latents = int((dummy_frames - 1) // 4) + 1
        self.gen_dummy = gen_dummy
        self.gen_dummy_only = gen_dummy_only
        self.use_dummy_image = use_dummy_image
        self.skip_frames = skip_frames
        self.verbose_messaging = verbose_messaging
        self.model = model
        self.secondary_model = secondary_model
        self.batches_script = batches_script
        self.g_output = {}

        disable_pbar = not utils.PROGRESS_BAR_ENABLED

        self.use_sigmas = self.sigmas

        self.guider = WarpedGuider_Basic()
        self.guider.set_model(self.model)
        callback = self.setup_callbacks()

        self.latent_window_size, self.batch_count, self.truncated_frame_count = self.get_latent_window_size(preferred_batch_size, self.preferred_frame_count, use_batch_size=use_batch_size)
        self.batch_frame_count = int(((self.truncated_frame_count - 1) // self.batch_count) + 1)
        print("latent_window_size: {}  |  batch_count: {}  |  truncated_frame_count: {}  | batch_frame_count: {}".format(self.latent_window_size, self.batch_count, self.truncated_frame_count, self.batch_frame_count))

        is_i2v = True
        is_v2v = False

        if start_image is None:
            start_image = torch.zeros([1, self.t2v_height, self.t2v_width, 3], dtype=torch.float32, device=self.offload_device)
            is_i2v = False
        else:
            if start_image.shape[0] > 1:
                is_v2v = True
                is_i2v = False

        if self.mode == "Wan 2.2 TI2V":
            self.latent_depth = 48
        else:
            self.latent_depth = 16

        print("start_image Shape: {}".format(start_image.shape))

        if self.mode != "Wan 2.2 TI2V":
            self.width = int(start_image.shape[2] // 8)
            self.height = int(start_image.shape[1] // 8)
        else:
            self.width = int(start_image.shape[2] // 16)
            self.height = int(start_image.shape[1] // 16)

        noise, dummy_noise = self.setup_latent_noise()

        output_images = []
        output_images_latents = None
        interrupted = False
        generation_status = ""
        total_images_generated = 0
        is_dummy_section = False
        is_first_section = False
        is_last_section = False
        switch_guider = True
        last_image = None
        positive_prompt = positive
        valid_output = False

        if not self.gen_dummy:
            print("Generating {} Batches".format(self.batch_count))
        else:
            if not self.gen_dummy_only:
                print("Generating Dummy Batch Plus {} Batches".format(self.batch_count))
            else:
                print("Generating Dummy Batch Only")

        generation_batches = list(range(self.batch_count))

        if self.gen_dummy and (not self.gen_dummy_only):
            generation_batches = [0] + generation_batches
            is_dummy_section = True
        else:
            generation_batches = [0]
            is_dummy_section = True

        print("generation_batches: {}".format(generation_batches))

        try:
            for i, generation_batch in enumerate(generation_batches):
                batch_number = generation_batches[i]
                batch_key = "{}".format(batch_number)

                if is_dummy_section:
                    print("Generating Dummy Batch.")
                    is_first_section = False

                    if not self.gen_dummy_only:
                        is_last_section = False
                    else:
                        is_last_section = True

                    if not batch_scripts is None:
                        positive_prompt = batch_scripts["dummy"]
                        print("\nDummy Section Positive Prompt: {}\n".format(positive_prompt))
                        last_prompt = positive_prompt
                else:
                    print("Generating Batch {}.".format(batch_number))

                    is_first_section = batch_number == min(generation_batches)
                    is_last_section = batch_number == max(generation_batches)

                    if not batch_scripts is None:
                        if batch_key in batch_scripts.keys():
                            positive_prompt = batch_scripts[batch_key]
                            last_prompt = positive_prompt
                        else:
                            positive_prompt = last_prompt

                        print("\nBatch: {} Positive Prompt: {}\n".format(batch_number, positive_prompt))

                print("is_first_section: {}  |  is_last_section: {}  |  is_dummy_section: {}".format(is_first_section, is_last_section, is_dummy_section))

                mm.soft_empty_cache()
                gc.collect()
                time.sleep(0.1)

                image_embeds = None

                print("start_image Shape: {}".format(start_image.shape))

                if is_i2v or is_v2v:
                    # print("GENERAL HERE 1")

                    if len(start_image.shape) < 4:
                        start_image = start_image.unsqueeze(0)

                    if mode != "Wan 2.2 TI2V":
                        print("Performing clip_vision_encode.")
                        image_embeds = self.clip_vision_encode(start_image.clone().detach())
                        print("Performing clip_vision_encode...Done!")

                if not is_dummy_section:
                    noise_latents = noise[batch_number]
                else:
                    noise_latents = dummy_noise

                print("noise_latents Shape: {}".format(noise_latents.shape))

                noise_mask = None

                if is_i2v:
                    if (not self.secondary_model is None) and switch_guider:
                        switch_guider = False
                        callback = None
                        self.guider = WarpedGuider_Basic()
                        self.guider.set_model(self.secondary_model)
                        callback = self.setup_callbacks()
                        self.use_clip = self.clip.clone()

                if mode == "Hunyuan":
                    if is_i2v or is_v2v:
                        temp_cond = self.hunyuan_text_encode(self.use_clip, image_embeds, positive_prompt, hunyuan_image_interleave)
                        positive_cond, latents = self.hunyuan_encode(temp_cond, self.vae, start_image.shape[2], start_image.shape[1], ((noise_latents.shape[2] - 1) * 4) + 1, 1, hunyuan_guidance_type, start_image=start_image)

                        if "noise_mask" in latents:
                            noise_mask = latents["noise_mask"] #* scaling_strength

                        if is_v2v:
                            scaling_strength = vae_scaling_factor

                        latents = latents["samples"] * scaling_strength
                    else:
                        positive_cond = self.do_text_encode(self.use_clip, positive_prompt)
                        latents = torch.zeros([1, 16, noise_latents.shape[2], self.height, self.width], dtype=torch.float32, device=self.offload_device)

                        latents = latents * scaling_strength

                    if use_flux_guidance:
                        positive_cond = self.apply_flux_guidance(positive_cond, flux_guidance)

                    self.guider.set_conds_single(positive_cond)
                else:
                    if not is_i2v:
                        # print("HERE WAN 1")
                        positive_cond, negative_cond = self.wan_text_encode(self.use_clip, positive_text=positive_prompt, negative_text=negative)
                        positive_cond, negative_cond, latents = self.wan_encode(positive_cond, negative_cond, self.vae, start_image.shape[2], start_image.shape[1], ((noise_latents.shape[2] - 1) * 4) + 1, 1, start_image=None, clip_vision_output=None)

                        if "noise_mask" in latents:
                            noise_mask = latents["noise_mask"] * scaling_strength

                        latents = latents["samples"] * scaling_strength
                    else:
                        # print("HERE WAN 2")

                        temp_cond, negative_cond = self.wan_text_encode(self.use_clip, positive_text=positive_prompt, negative_text=negative)
                        positive_cond, negative_cond, latents = self.wan_encode(temp_cond, negative_cond, self.vae, start_image.shape[2], start_image.shape[1], ((noise_latents.shape[2] - 1) * 4) + 1, 1, start_image=start_image, clip_vision_output=image_embeds)

                        if "noise_mask" in latents:
                            noise_mask = latents["noise_mask"] * scaling_strength

                        latents = latents["samples"] * scaling_strength

                    self.guider.set_conds_both(positive_cond, negative_cond)


                print("latents Shape: {}".format(latents.shape))

                num_frames = int(((latents.shape[2] - 1) * 4) + 1)

                print("\nDecoded Width is {}  |  Decoded Height is {}".format(int(self.width * 8), int(self.height * 8)))

                print("-------------------------------------------------------------------------------------------")
                print("WarpedSamplerCustomScripted: Latents Shape: {}  |  Noise Latents Shape: {}".format(latents.shape, noise_latents.shape))
                print("-------------------------------------------------------------------------------------------")

                print("Noise Shape: {}  |  Latents Shape: {}".format(noise_latents.shape, latents.shape))
                print("WarpedSamplerCustomScripted: Generating {} Frames in {} Latents....".format(num_frames, latents.shape[2]))

                samples = self.guider.sample(noise_latents, latents, self.sampler, self.use_sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=self.seed)
                samples = samples.to(mm.intermediate_device())

                mm.unload_all_models()
                mm.soft_empty_cache()
                gc.collect()
                time.sleep(1)

                if len(samples.shape) < 5:
                    samples = samples.unsqueeze(0)

                samples = samples.clone().detach() / scaling_strength # (scaling_strength * vae_scaling_factor)

                print("WarpedSamplerCustomScripted: Decoding Latents...")
                decoded_images = self.decode_tiled(samples)

                if not is_dummy_section or (is_dummy_section and ((self.use_dummy_image == "all") or self.gen_dummy_only)):
                    output_images.append(decoded_images)
                    total_images_generated += decoded_images.shape[0]
                    valid_output = True

                samples = None

                # output_images = self.process_skip_images(output_images, skip_frames)
                print("WarpedSamplerCustomScripted: Decoded Images Shape: {}".format(decoded_images.shape[0]))

                # if output_latents:
                #     output_images_latents = self.encode_tiled(output_images)

                print("WarpedSamplerCustomScripted: Generating {} frames in {} latents...Done.".format(num_frames, latents.shape[2]))

                if not is_i2v and not is_v2v:
                    if not self.secondary_model is None:
                        self.guider.model_patcher.model.to(get_offload_device())

                        if not self.secondary_sigmas is None:
                            self.use_sigmas = self.secondary_sigmas

                if not is_last_section:
                    decoded_tuple = torch.split(decoded_images, 1, dim=0)
                    decoded_split = [item for item in decoded_tuple]
                    decoded_images = decoded_images.to(get_offload_device())
                    decoded_images = None

                    if (not is_dummy_section) or (is_dummy_section and ((self.use_dummy_image == "last") or (self.use_dummy_image == "all"))):
                        start_image = decoded_split[len(decoded_split) - 1].clone().detach()
                    elif is_dummy_section:
                        if self.use_dummy_image == "first":
                            start_image = decoded_split[0].clone().detach()
                        elif self.use_dummy_image == "middle":
                            start_image = decoded_split[int(len(decoded_split) // 2)].clone().detach()
                        else:
                            random.seed(self.seed)
                            start_image = dummy_split[random.randrange(0, len(decoded_split) - 1, 1)].clone().detach()

                    if not is_v2v:
                        is_i2v = True

                if last_image is None:
                    last_image = start_image.clone().detach()

                samples = None
                latents = None

                mm.soft_empty_cache()
                gc.collect()
                time.sleep(0.1)

                print("*******************************************")
                print("****** WarpedSamplerCustomScripted: Total Images Generated {}".format(total_images_generated))
                print("*******************************************\n")

                generation_status = "****** WarpedSamplerCustomScripted: Total Images Generated {} ******".format(total_images_generated)

                interrupted = False

                if is_dummy_section:
                    if self.gen_dummy_only:
                        break

                    is_dummy_section = False
        except mm.InterruptProcessingException as ie:
            interrupted = True
            print(f"\nWarpedSamplerCustomScripted: Processing Interrupted.")
            print("WarpedSamplerCustomScripted: Returning only partial results (if any).\n If zero images generated, a blank yellow image will be returned, and valid_output will return False.\n")
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

            generation_status = f"\nWarpedSamplerCustomScripted: Processing Interrupted."

            traceback.print_tb(ie.__traceback__, limit=99, file=sys.stdout)

            # raise mm.InterruptProcessingException(f"WarpedSamplerCustomScripted: Processing Interrupted.")
            pass

        except BaseException as e:
            print(f"\nWarpedSamplerCustomScripted: Exception During Processing: {str(e)}")
            print("WarpedSamplerCustomScripted: Returning only partial results (if any).\n If zero images generated, a blank red image will be returned, and valid_output will return False.\n")
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            time.sleep(1)

            generation_status = f"WarpedSamplerCustomScripted: Exception During Processing: {str(e)}"
            generation_status = "{}{}".format(generation_status, "WarpedSamplerCustomScripted: Returning only partial results (if any).\nIf zero images generated, a blank red image will be returned, and valid_output will return False.")

            traceback.print_tb(e.__traceback__, limit=99, file=sys.stdout)

            pass

        callback = None
        self.guider.model_patcher.model.to(get_offload_device())

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

        if interrupted and not valid_output:
            temp_image = Image.new('RGB', (self.width * 8, self.height * 8), color = 'yellow')
            image = pil2tensorSwap(temp_image)
            final_images = image
        elif not valid_output:
            temp_image = Image.new('RGB', (self.width * 8, self.height * 8), color = 'red')
            image = pil2tensorSwap(temp_image)
            final_images = image
        else:
            final_images = self.assemble_final_result(output_images)

        if output_images_latents is None:
            output_images_latents = torch.zeros([1, 16, 1, self.height, self.width], dtype=torch.float32, device=self.offload_device)

        return (final_images, {"samples": output_images_latents}, self.seed, generation_status, last_image, valid_output,)

    def assemble_final_result(self, image_batches):
        if self.blend_frames < 1:
            resulting_images = None
            for entry in image_batches:
                if not resulting_images is None:
                    resulting_images = torch.cat((resulting_images, entry), 0)
                else:
                    resulting_images = entry

                entry.to(device=self.offload_device)
                entry = None
        else:
            blend_value = 1.0 / self.blend_frames
            i = 0
            while i < (len(image_batches) - 1):
                alpha_blend_val = blend_value
                blend_count = self.blend_frames

                image_batches_tuple = torch.split(image_batches[i], 1, dim=0)
                image_batches_split = [item for item in image_batches_tuple]
                image1 = image_batches_split[len(image_batches_split) - 1]
                # image1 = image_batches_split[len(image_batches_split) - 4]
                image_batches_tuple = None
                image_batches_split = None

                image_batches_tuple = torch.split(image_batches[i + 1], 1, dim=0)
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

                image_batches_tuple = torch.split(image_batches[i], image_batches[i].shape[0] - 3, dim=0)
                image_batches_split = [item for item in image_batches_tuple]
                image_batches_tuple = None

                image_batches[i] = torch.cat((image_batches_split[0], blend_latents), 0)
                blend_latents = None
                image_batches_split = None

                self.cleanup(unload_models=False, cleanup_models=False, cleanup_cuda=True)

                i += 1

            resulting_images = None
            for entry in image_batches:
                if not resulting_images is None:
                    resulting_images = torch.cat((resulting_images, entry), 0)
                else:
                    resulting_images = entry

                entry.to(device=self.offload_device)

            image_batches = None

            self.cleanup(unload_models=False, cleanup_models=False, cleanup_cuda=True)


        print("assemble_final_result: Full decoded images count: {}".format(resulting_images.shape[0]))

        if self.skip_frames < 1:
            return resulting_images

        skipped_frames = 1

        image_batches_tuple = torch.split(resulting_images, 1, dim=0)
        image_batches_split = [item for item in image_batches_tuple]

        resulting_images = None

        for image in image_batches_split:
            if skipped_frames <= self.skip_frames:
                skipped_frames += 1
                continue

            if not resulting_images is None:
                resulting_images = torch.cat((resulting_images, image), 0)
            else:
                resulting_images = image

        print("assemble_final_result: Final decoded images count: {}".format(resulting_images.shape[0]))

        return resulting_images

    def cleanup(self, unload_models=False, cleanup_models=False, cleanup_cuda=False):
        if unload_models:
            mm.unload_all_models()

        if cleanup_models:
            mm.cleanup_models()

        if cleanup_cuda:
            mm.soft_empty_cache()

        gc.collect()
        time.sleep(1)

        return

    def generate_noise(self, input_latent, generator=None):
        latent_image = input_latent["samples"]
        return warped_prepare_noise(latent_image, self.seed, generator=generator)

    def setup_latent_noise(self):
        dummy_noise = None

        total_noise_frames = 0
        rnd = torch.Generator("cpu").manual_seed(self.seed)

        if self.gen_dummy:
            total_noise_frames += self.dummy_latents

        total_noise_frames += (self.latent_window_size * self.batch_count) - self.batch_count + 1

        temp_noise = torch.randn((1, self.latent_depth, total_noise_frames, self.height, self.width), generator=rnd, device=rnd.device).to(device=rnd.device, dtype=torch.float32)

        noise = []

        latent_batches_tuple = torch.split(temp_noise, self.dummy_latents, dim=2)
        latent_batches_split = [item for item in latent_batches_tuple]

        dummy_noise = latent_batches_split[0].clone().detach()

        temp_noise = None

        i = 0

        for entry in latent_batches_split:
            if i != 0:
                if not temp_noise is None:
                    temp_noise = torch.cat((temp_noise, entry.clone().detach()), 2)
                else:
                    temp_noise = entry.clone().detach()

            i += 1

        latent_batches_tuple = None
        latent_batches_tuple = None

        latent_batches_tuple = torch.split(temp_noise, self.latent_window_size - 1, dim=2)
        latent_batches_split = [item for item in latent_batches_tuple]
        latent_batches_tuple = None

        for entry in latent_batches_split:
            noise.append(entry.clone().detach())

        noise[len(noise) - 2] = torch.cat((noise[len(noise) - 2], noise[len(noise) - 1]), 2)
        del noise[-1]

        latent_batches_split = None

        if self.verbose_messaging:
            if not dummy_noise is None:
                print("setup_latent_noise: Dummy Noise Shape: {}".format(dummy_noise.shape))

            i = 0
            for entry in noise:
                print("setup_latent_noise: Batch: {}  |  noise Shape: {}".format(i, entry.shape))
                i += 1

            print("\n")

        return noise, dummy_noise

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

        if self.verbose_messaging:
            print("get_latent_window_size: Next Lowest Initialized To: {}".format(next_lowest))
            print("get_latent_window_size: Next Highest Initialized To: {}".format(next_highest))

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

        print("Unable to find a compatible latent_window_size for number of frames = {} and preferred_batch_size = {}.".format(frame_count, preferred_batch_size))
        print("Recalculating Number Of Frames Based On preferred_batch_size of: {}".format(preferred_batch_size))

        return self.calculate_new_number_of_frames(preferred_batch_size, (((frame_count - 1) // 4) * 4) + 1, use_batch_size)

    def calculate_new_number_of_frames(self, preferred_batch_size, frame_count, use_batch_size):
        working_batch_size = preferred_batch_size - 1
        working_frame_count = frame_count - 1

        next_lowest = next_highest = working_frame_count
        next_lowest_found = False
        next_highest_found = False

        while next_lowest > 37:
            next_lowest -= 4

            if int(next_lowest % working_batch_size) == 0:
                next_lowest_found = True
                break

        if next_lowest_found and (use_batch_size == "next_lowest"):
            return self.decoded_to_encoded_length(preferred_batch_size), int(next_lowest // working_batch_size), next_lowest + 1

        while next_highest < 999997:
            next_highest += 4

            if int(next_highest % working_batch_size) == 0:
                next_highest_found = True
                break

        if next_highest_found and (use_batch_size == "next_highest"):
            return self.decoded_to_encoded_length(preferred_batch_size), int(next_highest // working_batch_size), next_highest + 1

        if next_lowest_found and next_highest_found:
            if (working_frame_count - next_lowest) <= (next_highest - working_frame_count):
                return self.decoded_to_encoded_length(preferred_batch_size), int(next_lowest // working_batch_size), next_lowest + 1

            return self.decoded_to_encoded_length(preferred_batch_size), int(next_highest // working_batch_size), next_highest + 1

        if next_lowest_found:
            return self.decoded_to_encoded_length(preferred_batch_size), int(next_lowest // working_batch_size), next_lowest + 1

        if next_highest_found:
            return self.decoded_to_encoded_length(preferred_batch_size), int(next_highest // working_batch_size), next_highest + 1

        raise ValueError("Unable to find a compatible latent_window_size for number of frames = {} and preferred_batch_size = {}.".format(frame_count, preferred_batch_size))

    def clip_vision_encode(self, image, crop="center"):
        # image_device = image.get_device()
        # image = image.to(device=self.device)

        crop_image = False
        if crop != "center":
            crop_image = False
        output = self.clip_vision.encode_image(image, crop=crop_image)

        # image = image.to(device=image_device)

        mm.soft_empty_cache()
        gc.collect()
        time.sleep(0.25)

        return output

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

    def apply_flux_guidance(self, conditioning, guidance):
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return c

    def hunyuan_text_encode(self, clip, clip_vision_output, prompt, image_interleave):
        tokens = clip.tokenize(prompt, llama_template=PROMPT_TEMPLATE_ENCODE_VIDEO_I2V, image_embeds=clip_vision_output.mm_projected, image_interleave=image_interleave)
        return clip.encode_from_tokens_scheduled(tokens)

    def hunyuan_encode(self, positive, vae, width, height, length, batch_size, guidance_type, start_image=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        out_latent = {}

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length, :, :, :3].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

            concat_latent_image = partial_encode_tiled(vae, start_image) # vae.encode(start_image)
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            if guidance_type == "v1 (concat)":
                cond = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            elif guidance_type == "v2 (replace)":
                cond = {'guiding_frame_index': 0}
                latent[:, :, :concat_latent_image.shape[2]] = concat_latent_image
                out_latent["noise_mask"] = mask
            elif guidance_type == "custom":
                cond = {"ref_latent": concat_latent_image}

            positive = node_helpers.conditioning_set_values(positive, cond)

        out_latent["samples"] = latent
        return positive, out_latent

    def wan_text_encode(self, clip, positive_text="", negative_text=""):
        print("wan_text_encode: Loading clip model to device: {}".format(clip.patcher.load_device))
        clip.patcher.model.to(device=clip.patcher.load_device)

        print("wan_text_encode: Encoding Prompts...")
        positive_conditioning = self.do_text_encode(clip, positive_text)
        negative_conditioning = self.do_text_encode(clip, negative_text)
        print("wan_text_encode: Encoding Prompts...Done.")

        print("wan_text_encode: Unloading clip model to device: {}".format(get_offload_device()))
        clip.patcher.model.to(device=get_offload_device())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        time.sleep(1)

        return positive_conditioning, negative_conditioning

    def wan_encode_image(self, vae, width, height, length, batch_size, start_image=None):
        latent = torch.zeros([1, self.latent_depth, ((length - 1) // 4) + 1, height // 16, width // 16], device=comfy.model_management.intermediate_device())

        if start_image is None:
            out_latent = {}
            out_latent["samples"] = latent
            return out_latent

        mask = torch.ones([latent.shape[0], 1, ((length - 1) // 4) + 1, latent.shape[-2], latent.shape[-1]], device=comfy.model_management.intermediate_device())

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            latent_temp = vae.encode(start_image)
            latent[:, :, :latent_temp.shape[-3]] = latent_temp
            mask[:, :, :latent_temp.shape[-3]] *= 0.0

        out_latent = {}
        latent_format = comfy.latent_formats.Wan22()
        latent = latent_format.process_out(latent) * mask + latent * (1.0 - mask)
        out_latent["samples"] = latent.repeat((batch_size, ) + (1,) * (latent.ndim - 1))
        out_latent["noise_mask"] = mask.repeat((batch_size, ) + (1,) * (mask.ndim - 1))

        return out_latent


    def encode(self, clip, text):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        tokens = clip.tokenize(text)
        return clip.encode_from_tokens_scheduled(tokens)

    def do_text_encode(self, clip, text):

        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        tokens = clip.tokenize(text)
        return_encoding = clip.encode_from_tokens_scheduled(tokens)

        return return_encoding

    def wan_encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None):
        if self.mode == "Wan 2.1" or self.mode == "Wan 2.2 Standard":
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
        else:
            out_latent = self.wan_encode_image(vae, width, height, length, batch_size, start_image)

        return positive, negative, out_latent

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
        print("WarpedSamplerCustomScripted: Encoded noise_latents_full Shape: {}".format(noise_latents_full.shape))

        if Decimal(self.noise_scale).compare(Decimal(0.00)) != 0:
            noise_latents_full = warped_prepare_noise(noise_latents_full, self.seed)
            print("WarpedSamplerCustomScripted: noise_latents_full Shape: {}".format(noise_latents_full.shape))

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

        print("WarpedSamplerCustomScripted: Encoded latents_full Shape: {}".format(latents.shape))
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

    CATEGORY = "Warped/General/Video"

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

    CATEGORY = "Warped/General/Video"

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

        print("WarpedBundleAllVideoImages: Width: {}  |  Height: {}".format(width, height))

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

    CATEGORY = "Warped/General/Image"

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

    CATEGORY = "Warped/Hunyuan/LeapFusion"

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
    CATEGORY = "Warped/General/Image/Animation"

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
        self.i2v_model = None

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

    def set_models(self, t2v_model, i2v_model):
        self.model_patcher = t2v_model
        self.model_options = t2v_model.model_options
        self.inner_executor = None
        self.i2v_model = i2v_model

        return

    def set_i2v_model(self):
        if not self.i2v_model is None:
            self.model_patcher = self.i2v_model
            self.model_options = self.i2v_model.model_options
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
    def set_conds_single(self, positive):
        self.inner_set_conds({"positive": positive})

    def set_conds_both(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

class WarpedBasicGuider:
    def __init__(self):
        self.guider = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL", ),
                    }
            }

    RETURN_TYPES = ("WARPED_GUIDER",)
    RETURN_NAMES = ("guider",)

    FUNCTION = "get_guider"
    CATEGORY = "Warped/General/Sampling/Guiders"

    def get_guider(self, model):
        self.guider = WarpedGuider_Basic()
        self.guider.set_model(model)
        return (self.guider,)

class WarpedDualGuider:
    def __init__(self):
        self.guider = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"t2v_model": ("MODEL", ),
                    "i2v_model": ("MODEL", ),
                    }
               }

    RETURN_TYPES = ("WARPED_GUIDER",)
    RETURN_NAMES = ("guider",)

    FUNCTION = "get_guider"
    CATEGORY = "Warped/General/Sampling/Guiders"

    def get_guider(self, t2v_model, i2v_model):
        self.guider = WarpedGuider_Basic()
        self.guider.set_models(t2v_model, i2v_model)

        return (self.guider,)

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

    CATEGORY = "Warped/General/Image"


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

    CATEGORY = "Warped/General/Video"

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
                    "starting_index": ("INT", {"default": 0, "min": 0, "max": 25000000, "step": 1}),
                    "num_frames": ("INT", {"default": 61, "min": 5, "max": 1000001, "step": 4}),
                    "use_gpu": ("BOOLEAN", {"default": True}),
                    },
                }
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", "INT", "INT", "FLOAT", )
    RETURN_NAMES = ("image_batch", "first_image", "last_image", "num_frames", "width", "height", "fps", )
    FUNCTION = "generate"

    CATEGORY = "Warped/General/Video"

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
        except:
            print("WarpedBundleVideoImages: Exception During Video File Read.")
        finally:
            cap.release()

        try:
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
            print("WarpedBundleVideoImages: Exception During Video File Processing.")

        batched_images = batched_images.to(get_offload_device())

        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        return (batched_images, first_image, last_image, int(batched_images.shape[0]), width, height, float(fps), )

class WarpedGetImageFromVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"video_path": ("STRING", {"default": ""}),
                    "image_index": ("INT", {"default": 0}),
                    },
                }
    RETURN_TYPES = ("IMAGE", "INT", "INT", )
    RETURN_NAMES = ("image", "width", "height", )
    FUNCTION = "generate"

    CATEGORY = "Warped/General/Video"

    def generate(self, video_path, image_index):
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('WarpedGetImageFromVideo: width = %d' % width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('WarpedGetImageFromVideo: height = %d' % height)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('WarpedGetImageFromVideo: length = %d' % length)

        return_image = None

        if image_index > length:
            image_index = length - 1

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, image_index)

            # Take each frame
            _, frameorig = cap.read()

            color_coverted = cv2.cvtColor(frameorig, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(color_coverted)
            return_image = pil2tensorSwap(pil_image, device=get_offload_device())

            if len(return_image.shape) < 4:
                return_image = return_image.unsqueeze(0)

            print("return_image: Shape: {}".format(return_image.shape))

            frameorig = None
            color_coverted = None
            pil_image = None
        except:
            print("WarpedGetImageFromVideo: Exception During Video File Read.")
        finally:
            cap.release()

        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        return (return_image, width, height, )

class WarpedGetTwoImagesFromVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"video_path": ("STRING", {"default": ""}),
                    "first_image_index": ("INT", {"default": 0}),
                    "second_image_index": ("INT", {"default": 1}),
                    },
                }
    RETURN_TYPES = ("IMAGE", "INT", "INT", )
    RETURN_NAMES = ("first_image", "second_image", "width", "height", )
    FUNCTION = "generate"

    CATEGORY = "Warped/General/Video"

    def generate(self, video_path, first_image_index, second_image_index):
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('WarpedGetImageFromVideo: width = %d' % width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('WarpedGetImageFromVideo: height = %d' % height)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('WarpedGetImageFromVideo: length = %d' % length)

        return_image_one = None
        return_image_two = None

        if first_image_index > length:
            first_image_index = 0

        if second_image_index > length:
            second_image_index = length - 1

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_image_index)

            # Take each frame
            _, frameorig = cap.read()

            color_coverted = cv2.cvtColor(frameorig, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(color_coverted)
            return_image_one = pil2tensorSwap(pil_image, device=get_offload_device())

            if len(return_image_one.shape) < 4:
                return_image_one = return_image_one.unsqueeze(0)

            print("return_image_one: Shape: {}".format(return_image_one.shape))

            cap.set(cv2.CAP_PROP_POS_FRAMES, second_image_index)

            # Take each frame
            _, frameorig = cap.read()

            color_coverted = cv2.cvtColor(frameorig, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(color_coverted)
            return_image_two = pil2tensorSwap(pil_image, device=get_offload_device())

            if len(return_image_two.shape) < 4:
                return_image_two = return_image_two.unsqueeze(0)

            print("return_image_two: Shape: {}".format(return_image_two.shape))

            frameorig = None
            color_coverted = None
            pil_image = None
        except:
            print("WarpedGetImageFromVideo: Exception During Video File Read.")
        finally:
            cap.release()

        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        return (return_image_one, return_image_two, width, height, )

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

    CATEGORY = "Warped/Wan/Conditioning"

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

    CATEGORY = "Warped/Hunyuan/Conditioning"

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

class WarpedImageFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                              "batch_index": ("INT", {"default": 0, "min": 0, "max": 4095}),
                              "length": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "frombatch"

    CATEGORY = "Warped/General/Image"

    def frombatch(self, image, batch_index, length):
        # print("\n-------------------------------------------------------------------------------")
        # print(image)
        # print("-------------------------------------------------------------------------------\n")

        s_in = image

        if isinstance(s_in, list):
            s_in = s_in[0]

        batch_index = min(s_in.shape[0] - 1, batch_index)
        length = min(s_in.shape[0] - batch_index, length)
        s = s_in[batch_index:batch_index + length].clone()
        return (s,)

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

            return (new_model, new_clip,)

        except Exception as e:
            logger.error(f"Error applying LoRA {lora_name}: {str(e)}")

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
    CATEGORY = "Warped/Wan/Lora"

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

    CATEGORY = "Warped/General/Image"

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

        newImage.to(device=get_offload_device())
        newImage = None
        final_image.to(device=get_offload_device())

        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        return (final_image, final_image.shape[2], final_image.shape[1],)

def get_base_lora_dirs():
    return folder_paths.get_folder_paths("loras")

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

    CATEGORY = "Warped/General/Lora"

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
    CATEGORY = "Warped/Hunyuan/Lora"
    OUTPUT_NODE = False
    DESCRIPTION = "LoRA, single blocks double blocks"

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

        diffusers_lora = convert_lora(lora, convert_to="diffusion_model")
        filtered_lora = filter_lora_keys(diffusers_lora, blocks_type)

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
    DESCRIPTION = "Select a Hunyuan LoRA models"

    def select_multiple_loras(self, **kwargs):
        loras_list = []

        prev_lora = kwargs.get(f"prev_lora")

        if prev_lora is not None:
            loras_list.extend(prev_lora)

        for i in range(1, 5):
            temp_lora_name = kwargs.get(f"lora_0{i}")
            temp_strength = kwargs.get(f"strength_0{i}")
            temp_fuse_lora = kwargs.get(f"fuse_lora_0{i}")

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

class WarpedFramepackMultiLoraSelectExt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora_01": (['None'] + folder_paths.get_filename_list("loras"), {"tooltip": "LORA models are expected to have .safetensors extension"}),
                "strength_01": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "blocks_01": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
                "fuse_lora_01": ("BOOLEAN", {"default": False, "tooltip": "Fuse the LORA model with the base model. This is recommended for better performance."}),
               "lora_02": (['None'] + folder_paths.get_filename_list("loras"), {"tooltip": "LORA models are expected to have .safetensors extension"}),
                "strength_02": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "blocks_02": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
                "fuse_lora_02": ("BOOLEAN", {"default": False, "tooltip": "Fuse the LORA model with the base model. This is recommended for better performance."}),
               "lora_03": (['None'] + folder_paths.get_filename_list("loras"), {"tooltip": "LORA models are expected to have .safetensors extension"}),
                "strength_03": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "blocks_03": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
                "fuse_lora_03": ("BOOLEAN", {"default": False, "tooltip": "Fuse the LORA model with the base model. This is recommended for better performance."}),
               "lora_04": (['None'] + folder_paths.get_filename_list("loras"), {"tooltip": "LORA models are expected to have .safetensors extension"}),
                "strength_04": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "blocks_04": (["all", "double_blocks", "single_blocks"], {"default": "all", "tooltip": "all, single only, or double only block."}),
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
    DESCRIPTION = "Select a Hunyuan LoRA models"

    def select_multiple_loras(self, **kwargs):
        loras_list = []

        prev_lora = kwargs.get(f"prev_lora")

        if prev_lora is not None:
            loras_list.extend(prev_lora)

        for i in range(1, 5):
            temp_lora_name = kwargs.get(f"lora_0{i}")
            temp_strength = kwargs.get(f"strength_0{i}")
            temp_blocks =  kwargs.get(f"blocks_0{i}")
            temp_fuse_lora = kwargs.get(f"fuse_lora_0{i}")

            if (temp_lora_name != "None") and (Decimal(temp_strength).compare(Decimal(0.0)) != 0):
                lora = {
                    "path": folder_paths.get_full_path("loras", temp_lora_name),
                    "strength": temp_strength,
                    "blocks": temp_blocks,
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
    CATEGORY = "Warped/Hunyuan/Lora"
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

        # # Get the full path to the LoRA file
        # lora_path = folder_paths.get_full_path("loras", lora_name)
        # if not os.path.exists(lora_path):
        #     raise ValueError(f"LoRA file not found: {lora_path}")

        # Load the LoRA weights
        # lora_weights = utils.load_torch_file(lora_path)
        lora_weights = warped_load_lora_weights(lora_name)
        lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
        new_weights = {}

        for key in lora_weights:
            new_weights[key] = torch.mul(lora_weights[key], strength).to(dtype=torch.bfloat16)

        # Filter the LoRA weights based on the block type
        filtered_lora = filter_lora_keys(new_weights, blocks_type)

        return new_weights, filtered_lora

    def load_multiple_loras(self, model, **kwargs):
        """Load and apply multiple LoRA models."""
        from comfy.sd import load_lora_for_models

        temp_lora_name = kwargs.get(f"lora_name")
        temp_strength = kwargs.get(f"strength")
        temp_blocks_type = kwargs.get(f"blocks_type")

        lora_metadata = []

        if not temp_lora_name is None and temp_strength != 0:
            print("\n**** Lora Name: {}  |  Strength: {}  |  Block Types: {}\n".format(temp_lora_name, temp_strength, temp_blocks_type))
            lora_metadata.append("{}".format("Lora: {} | Strength: {} | Block Types: {}".format(temp_lora_name, temp_strength, temp_blocks_type)))

            lora_weights, filtered_lora = self.load_lora(temp_lora_name, temp_strength, temp_blocks_type)

            # Apply the LoRA weights to the model
            if filtered_lora:
                model, _ = load_lora_for_models(model, None, filtered_lora, 1.0, 0)
            else:
                model, _ = load_lora_for_models(model, None, lora_weights, 1.0, 0)

        for i in range(1, 5):
            temp_lora_name = kwargs.get(f"lora_0{i}")
            temp_strength = kwargs.get(f"strength_0{i}")
            temp_blocks_type = kwargs.get(f"blocks_type_0{i}")

            if temp_lora_name != "None" and temp_strength != 0:
                temp_message = "{}".format("Lora: {} | Strength: {} | Block Types: {}".format(temp_lora_name, temp_strength, temp_blocks_type))
                print(temp_message)

                lora_metadata.append(temp_message)

                # Load and filter the LoRA weights
                lora_weights, filtered_lora = self.load_lora(temp_lora_name, temp_strength, temp_blocks_type)

                # Apply the LoRA weights to the model
                if filtered_lora:
                    model, _ = load_lora_for_models(model, None, filtered_lora, 1.0, 0)
                else:
                    model, _ = load_lora_for_models(model, None, lora_weights, 1.0, 0)

        return (model, lora_metadata, )

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return f"{kwargs.get('lora_name')}_{kwargs.get('strength')}_{kwargs.get('blocks_type')}_" \
               f"{kwargs.get('lora_01')}_{kwargs.get('strength_01')}_{kwargs.get('blocks_type_01')}_" \
               f"{kwargs.get('lora_02')}_{kwargs.get('strength_02')}_{kwargs.get('blocks_type_02')}_" \
               f"{kwargs.get('lora_03')}_{kwargs.get('strength_03')}_{kwargs.get('blocks_type_03')}_" \
               f"{kwargs.get('lora_04')}_{kwargs.get('strength_04')}_{kwargs.get('blocks_type_04')}"

class WarpedMultiLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_01": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_model_01": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip_01": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
                "lora_02": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_model_02": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip_02": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
                "lora_03": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_model_03": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip_03": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
                "lora_04": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_model_04": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip_04": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            },
          "optional": {
                "lora_name": ("STRING", {"default": None, "forceInput": True}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", )
    RETURN_NAMES = ("model", "clip", "lora_metadata",)
    OUTPUT_IS_LIST = (False, False, True)
    FUNCTION = "load_multiple_loras"
    CATEGORY = "Warped/General/Loaders"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths. Model input is required."

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return model, clip

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_path = os.path.abspath(lora_path)

        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

        return model_lora, clip_lora

    def load_multiple_loras(self, model, clip, **kwargs):
        """Load and apply multiple LoRA models."""
        from comfy.sd import load_lora_for_models

        temp_lora_name = kwargs.get(f"lora_name")
        temp_strength_model = kwargs.get(f"strength_model")
        temp_strength_clip = kwargs.get(f"strength_clip")

        temp_model = model
        temp_clip = clip

        lora_metadata = []

        if not temp_lora_name is None and ((temp_strength_model != 0) or (temp_strength_clip != 0)):
            print("Lora Name: {}  |  Strength Model: {}  |  Strength Clip: {}".format(temp_lora_name, temp_strength_model, temp_strength_clip))
            lora_metadata.append("Lora: {} | Strength Model: {}  |  Strength clip: {}".format(temp_lora_name, temp_strength_model, temp_strength_clip))

            temp_model, temp_clip = self.load_lora(temp_model, temp_clip, temp_lora_name, temp_strength_model, temp_strength_clip)

        for i in range(1, 5):
            temp_lora_name = kwargs.get(f"lora_0{i}")

            if temp_lora_name == "None":
                continue

            temp_strength_model = kwargs.get(f"strength_model_0{i}")
            temp_strength_clip = kwargs.get(f"strength_clip_0{i}")

            print("Lora Name: {}  |  Strength Model: {}  |  Strength Clip: {}".format(temp_lora_name, temp_strength_model, temp_strength_clip))

            if not temp_lora_name is None and ((temp_strength_model != 0) or (temp_strength_clip != 0)):
                lora_metadata.append("Lora: {} | Strength Model: {}  |  Strength clip: {}".format(temp_lora_name, temp_strength_model, temp_strength_clip))
                # Load LoRA weights
                temp_model, temp_clip = self.load_lora(temp_model, temp_clip, temp_lora_name, temp_strength_model, temp_strength_clip)

        return (temp_model, temp_clip, lora_metadata, )

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
    CATEGORY = "Warped/Hunyuan/Merge"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

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
        lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")
        filtered_lora = filter_lora_keys(lora_weights, blocks_type)

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

        new_lora = {}

        for lora_key in temp_loras.keys():
            for key in temp_loras[lora_key]["filtered_lora"].keys():
                if not key in new_lora.keys():
                    new_lora[key] = None

        # Merge The Weighted Key Weights
        for key in new_lora.keys():
            for lora_key in temp_loras.keys():
                if key in temp_loras[lora_key]["filtered_lora"].keys():
                    if not new_lora[key] is None:
                        temp_weights = torch.mul(temp_loras[lora_key]["filtered_lora"][key], temp_loras[lora_key]["strength"])

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
                        new_lora[key] = torch.mul(temp_loras[lora_key]["filtered_lora"][key], temp_loras[lora_key]["strength"])

        if not save_metadata:
            metadata = None

        full_output_path, filename, counter, subfolder, filename_prefix = get_save_lora_path(model_prefix, self.base_output_dir)

        output_filename = os.path.join(full_output_path, "{}_{:05}.safetensors".format(model_prefix, counter))
        utils.save_torch_file(new_lora, output_filename, metadata=metadata)

        save_message = "Weights Saved To: {}".format(output_filename)
        print(save_message)

        return {"ui": {"tags": [save_message]}}

class WarpedHunyuanLoraConvertToMusubi:
    def __init__(self):
        self.base_output_dir = get_default_output_folder()
        os.makedirs(self.base_output_dir, exist_ok = True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_path": ("STRING", {"default": get_default_output_path()}),
                "lora": (get_lora_list(),),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert_lora"
    CATEGORY = "Warped/Hunyuan/Lora/Experimental"
    DESCRIPTION = "Convert Keys For Hunyuan LORA and Save Modified LORA."

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

    def convert_lora(self, save_path, lora, save_metadata=True):
        metadata = {"original_lora": "{}".format(lora)}

        # Load the LoRA weights
        temp_lora = self.load_lora(lora)
        new_lora = convert_to_musubi(temp_lora)

        if not save_metadata:
            metadata = None

        if len(new_lora) < 1:
            utils.save_torch_file(temp_lora, save_path, metadata=metadata)
        else:
            utils.save_torch_file(new_lora, save_path, metadata=metadata)

        save_message = "Weights Saved To: {}".format(save_path)

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
    CATEGORY = "Warped/Hunyuan/Lora"
    DESCRIPTION = "Convert Keys For Hunyuan LORA and Save Modified LORA."

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
        new_lora = convert_lora(temp_lora, convert_to=convert_to)

        if not save_metadata:
            metadata = None

        if len(new_lora) < 1:
            utils.save_torch_file(temp_lora, save_path, metadata=metadata)
        else:
            utils.save_torch_file(new_lora, save_path, metadata=metadata)

        save_message = "Weights Saved To: {}".format(save_path)

        return {"ui": {"tags": [save_message]}}

# class WarpedHunyuanLoraConvertKeys2:
#     def __init__(self):
#         self.base_output_dir = get_default_output_folder()
#         os.makedirs(self.base_output_dir, exist_ok = True)
#
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "save_path": ("STRING", {"default": get_default_output_path()}),
#                 "lora": (get_lora_list(),),
#                 "convert_to": (["diffusion_model", "transformer", "framepack"], {"default": "diffusion_model"}),
#                 "save_metadata": ("BOOLEAN", {"default": True}),
#             },
#         }
#
#     RETURN_TYPES = ()
#     OUTPUT_NODE = True
#     OUTPUT_IS_LIST = (True,)
#     FUNCTION = "convert_lora"
#     CATEGORY = "Warped/Hunyuan/Lora"
#     DESCRIPTION = "Convert Keys For Hunyuan LORA and Save Modified LORA."
#
#     def load_lora(self, lora_name: str) -> Tuple[Dict[str, torch.Tensor],]:
#         """Load and filter a single LoRA model."""
#         if not lora_name:
#             return {}
#
#         # Get the full path to the LoRA file
#         lora_path = folder_paths.get_full_path("loras", lora_name)
#         if not os.path.exists(lora_path):
#             raise ValueError(f"LoRA file not found: {lora_path}")
#
#         # Load the LoRA weights
#         lora_weights = utils.load_torch_file(lora_path)
#
#         return lora_weights
#
#     def convert_lora(self, save_path, lora, convert_to, save_metadata=True):
#         metadata = {"original_lora": "{}".format(lora)}
#
#         # Load the LoRA weights
#         temp_lora = self.load_lora(lora)
#
#         if convert_to != "framepack":
#             new_lora = convert_lora(temp_lora, convert_to=convert_to)
#         else:
#             new_lora = convert_lora(temp_lora, convert_to="diffusion_model")
#
#         for key in new_lora.keys():
#             print("LORA Key: {}".format(key))
#
#         if not save_metadata:
#             metadata = None
#
#         if len(new_lora) < 1:
#             utils.save_torch_file(temp_lora, save_path, metadata=metadata)
#         else:
#             utils.save_torch_file(new_lora, save_path, metadata=metadata)
#
#         save_message = "Weights Saved To: {}".format(save_path)
#
#         return {"ui": {"tags": [save_message]}}

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
    CATEGORY = "Warped/General/Lora"
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

            try:
                return metadata["__metadata__"]
            except:
                pass

        return {"metadata": "No Metadata" }

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

        lora_keys = []
        for key in lora_weights.keys():
            lora_keys.append("{}  |  Shape: {}\n".format(key, lora_weights[key].shape))
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
    CATEGORY = "Warped/Hunyuan/Lora"
    DESCRIPTION = "Convert Hunyuan Lora."

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
            lora_weights = convert_lora(lora_weights, convert_to="diffusion_model")

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
            temp_weights = lora_weights[key].clone().detach().to(dtype=torch.bfloat16)

            if temp_weights.shape[0] < temp_weights.shape[1]:
                padding = torch.zeros([target_res, temp_weights.shape[1]]).to(dtype=torch.bfloat16)

                # if upscale
                if temp_weights.shape[0] < target_res:
                    padding[:temp_weights.shape[0],:] = temp_weights
                    new_lora[key] = padding
                # if downscale
                else:
                    padding[:target_res,:] = temp_weights[:target_res,:]
                    new_lora[key] = padding
            else:
                padding = torch.zeros([temp_weights.shape[0], target_res]).to(dtype=torch.bfloat16)

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
                "cache_mode": (["disabled", "use_teacache", "use_magcache"], {"default": "use_magcache", "tooltip": "Whether or not to use magcache or teacache for faster sampling."}),
                "teacache_rel_l1_thresh": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The threshold for the relative L1 loss."}),
                "magcache_thresh": ("FLOAT", {"default": 0.1250, "min": 0.0000, "max": 0.3000, "step": 0.0010, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."}),
                "magcache_retention_ratio": ("FLOAT", {"default": 0.225, "min": 0.100, "max": 0.300, "step": 0.001, "tooltip": "The start percentage of the steps that will apply MagCache."}),
                "magcache_K": ("INT", {"default": 3, "min": 0, "max": 6, "step": 1, "tooltip": "The maxium skip steps of MagCache."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 32.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "preferred_frame_count": ("INT", {"default": 301, "min": 33, "max": 1000001, "step": 4, "tooltip": "For I2V and T2V, The total frames in the video. Disreguarded for V2V"}),
                "preferred_batch_size": ("INT", {"default": 61, "min": 17, "max": 161, "step": 4, "tooltip": "The preferred number of frames to use for sampling."}),
                "use_batch_size": (["next_lowest", "next_highest", "closest", "exact"], {"default": "next_lowest", "tooltip": "Number of frames generated may be impacted by choice."}),
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
                "start_embed_strength": ("FLOAT", {"default": 1.20, "min": 0.00, "max": 2.00, "step": 0.01, "tooltip": "Weighted average constant for image embed interpolation. If end image is not set, the embed's strength won't be affected"}),
                "secondary_embed_strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01, "tooltip": "Weighted average constant for image embed interpolation. If end image is not set, the embed's strength won't be affected"}),
                "video_image_batch": ("IMAGE", {"tooltip": "init Latents to use for video2video"} ),
                "denoise_strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "noise_strength": ("FLOAT", {"default": 1.00, "min": 0.10, "max": 1.00, "step": 0.01}),
                "blend_frames": ("INT", {"default":5, "min":0, "max": 16, "step": 1}),
                "t2v_width": ("INT", {"default":480, "min":256, "max": 1280, "step": 16}),
                "t2v_height": ("INT", {"default":720, "min":256, "max": 1280, "step": 16}),
                "dummy_frames": ("INT", {"default":17, "min":17, "max": 161, "step": 4, "tooltip": "Number of frames to generate in dummy batch."}),
                "gen_dummy": ("BOOLEAN", {"default": False, "tooltip": "For t2v or i2v only. Will generate a dummy batch to obtain a starting image for main generation."}),
                "gen_dummy_only": ("BOOLEAN", {"default": False, "tooltip": "Will generate dummy batch only."}),
                "dummy_cache_mode": (["disabled", "use_teacache", "use_magcache"], {"default": "use_magcache", "tooltip": "Whether or not to use magcache or teacache on dummy generation for faster sampling."}),
                "use_dummy_image": (["first", "middle", "last", "random", "all"], {"default": "last", "tooltip": "Which dummy batch image to start main generation."}),
                "v2v_context_count": ("INT", {"default":5, "min":3, "max": 10, "step": 1}),
                "verbose_messaging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "BOOLEAN",)
    RETURN_NAMES = ("images", "generation_status", "seed", "valid_output",)
    FUNCTION = "process"
    CATEGORY = "Warped/Framepack/Sampling"

    def process(self, model, vae, shift, positive, negative, preferred_frame_count, preferred_batch_size, use_batch_size, cache_mode, teacache_rel_l1_thresh,
                magcache_thresh, magcache_retention_ratio, magcache_K, steps, cfg,
                guidance_scale, seed, sampler, dec_tile_size, dec_overlap, dec_temporal_size, dec_temporal_overlap, skip_frames, clip_vision_model,
                gpu_memory_preservation, start_image=None, end_image=None, embed_interpolation="linear", start_embed_strength=1.0, secondary_embed_strength=1.0, video_image_batch=None,
                denoise_strength=1.00, noise_strength=1.00, blend_frames=0, t2v_width=640, t2v_height=640, dummy_frames=5, gen_dummy=False, gen_dummy_only=False, dummy_cache_mode="disabled", use_dummy_image="last",
                v2v_context_count=5, verbose_messaging=False):
        self.dec_tile_size = dec_tile_size
        self.dec_overlap = dec_overlap
        self.dec_temporal_size = dec_temporal_size
        self.dec_temporal_overlap = dec_temporal_overlap
        self.skip_frames = skip_frames
        self.vae = vae
        self.preferred_frame_count = preferred_frame_count
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.use_teacache = False
        self.use_magcache = False
        self.teacache_rel_l1_thresh = teacache_rel_l1_thresh
        self.magcache_thresh = magcache_thresh
        self.magcache_retention_ratio = magcache_retention_ratio
        self.magcache_K = magcache_K
        self.guidance_scale = guidance_scale
        self.sampler = sampler
        self.skip_frames = skip_frames
        self.gpu_memory_preservation = gpu_memory_preservation
        self.embed_interpolation = embed_interpolation
        self.start_embed_strength = start_embed_strength
        self.secondary_embed_strength = secondary_embed_strength
        self.denoise_strength = denoise_strength
        self.noise_strength = noise_strength
        self.transformer = model["transformer"]
        self.base_dtype = model["dtype"]
        self.positive = positive
        self.negative = negative
        self.device = mm.get_torch_device()
        self.offload_device = get_offload_device()
        self.clip_vision = clip_vision_model
        self.blend_frames = blend_frames
        self.shift = shift
        self.t2v_width=t2v_width
        self.t2v_height=t2v_height
        self.dummy_latents = int((dummy_frames - 1) // 4) + 1
        self.gen_dummy = gen_dummy
        self.gen_dummy_only = gen_dummy_only
        self.dummy_teacache = False
        self.dummy_magcache = False
        self.use_dummy_image = use_dummy_image
        self.v2v_context_count = v2v_context_count
        self.verbose_messaging = verbose_messaging

        if cache_mode == "use_teacache":
            self.use_teacache = True

        if cache_mode == "use_magcache":
            self.use_magcache = True

        if dummy_cache_mode == "use_teacache":
            self.dummy_teacache = True

        if dummy_cache_mode == "use_magcache":
            self.dummy_magcache = True

        print("Device: {}  | Offload Device: {}".format(self.device, self.offload_device))

        self.llama_vec = self.positive[0][0].to(self.base_dtype).to(self.device)
        self.clip_l_pooler = self.positive[0][1]["pooled_output"].to(self.base_dtype).to(self.device)

        if not math.isclose(self.cfg, 1.0):
            self.llama_vec_n = self.negative[0][0].to(self.base_dtype)
            self.clip_l_pooler_n = self.negative[0][1]["pooled_output"].to(self.base_dtype).to(self.device)
        else:
            self.llama_vec_n = torch.zeros_like(self.llama_vec, device=self.device)
            self.clip_l_pooler_n = torch.zeros_like(self.clip_l_pooler, device=self.device)

        self.llama_vec, self.llama_attention_mask = crop_or_pad_yield_mask(self.llama_vec, length=512)
        self.llama_vec_n, self.llama_attention_mask_n = crop_or_pad_yield_mask(self.llama_vec_n, length=512)

        if not video_image_batch is None:
            if len(video_image_batch.shape) < 4:
                video_image_batch = video_image_batch.unsqueeze(0)

            self.preferred_frame_count = video_image_batch.shape[0]

            self.latent_window_size, self.batch_count, truncated_frame_count = self.get_latent_window_size(preferred_batch_size, self.preferred_frame_count, use_batch_size=use_batch_size)

            if video_image_batch.shape[0] != truncated_frame_count:
                image_batches_tuple = torch.split(video_image_batch, truncated_frame_count, dim=0)
                image_batches_split = [item for item in image_batches_tuple]
                video_image_batch = image_batches_split[0]
        else:
            print("preferred_frame_count: {}".format(self.preferred_frame_count))
            self.latent_window_size, self.batch_count, truncated_frame_count = self.get_latent_window_size(preferred_batch_size, self.preferred_frame_count, use_batch_size=use_batch_size)

        self.num_frames = self.encoded_to_decoded_length(self.latent_window_size)
        self.total_frames = truncated_frame_count

        if self.verbose_messaging:
            print("num_frames: {}".format(self.num_frames))

        self.cleanup(unload_models=True, cleanup_models=True, cleanup_cuda=True)

        if not video_image_batch is None:
            self.mode = VideoGenerationType.V2V

            return self.process_v2v(video_image_batch)

        if not start_image is None:
            self.mode = VideoGenerationType.I2V
            return self.process_i2v(start_image, end_image)

        self.mode = VideoGenerationType.T2V

        return self.process_t2v()

    def process_t2v(self):
        return self.process_i2v(None, None)

    def process_i2v(self, start_image, end_image):
        start_latent = None
        end_latent = None

        if self.mode == VideoGenerationType.I2V:
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
            print("process_i2v: Image Height: {}".format(self.height))
            print("process_i2v: total_frames: {}".format(self.total_frames))
            print("process_i2v: batch_count: {}".format(self.batch_count))
            print("process_i2v: latent_window_size: {}".format(self.latent_window_size))

        start_image_encoder_last_hidden_state = image_embeds["last_hidden_state"].to(self.base_dtype).to(self.device)

        if has_end_image:
            assert end_image_embeds is not None
            end_image_encoder_last_hidden_state = end_image_embeds["last_hidden_state"].to(self.base_dtype).to(self.device)
        else:
            end_image_encoder_last_hidden_state = torch.zeros_like(start_image_encoder_last_hidden_state)

        rnd = torch.Generator("cpu").manual_seed(self.seed)
        total_generated_latent_frames = 0

        if self.mode == VideoGenerationType.I2V:
            if start_latent is None:
                raise ValueError("A start_image value is required for I2V.")

            cat_list = []

            history_latents = torch.zeros(size=(1, 16, 19, H, W), dtype=torch.float32).cpu()
            original_history_latents = history_latents.clone().detach()

            cat_list.append(history_latents)
            cat_list.append(start_latent.to(history_latents))

            history_latents = torch.cat(cat_list, dim=2)
        else:
            history_latents = torch.zeros(size=(1, 16, 20, H, W), dtype=torch.float32).cpu()
            original_history_latents = torch.zeros(size=(1, 16, 19, H, W), dtype=torch.float32).cpu()

        latent_paddings_list = list(range(total_latent_sections))
        latent_paddings = latent_paddings_list.copy()  # Create a copy for iteration

        comfy_model, patcher, callback = self.initialize_comfy_model()

        latent_batches_gend = []

        is_dummy_section = False
        was_dummy_used = False

        noise, dummy_noise = self.setup_latent_noise(self.latent_window_size * self.batch_count, W, H)

        if self.gen_dummy:
            # dummy_noise = self.setup_dummy_noise(W, H)

            if not self.gen_dummy_only:
                latent_paddings.insert(0, 0)
            else:
                latent_paddings = [0]
                latent_paddings_list = latent_paddings.copy()
                total_latent_sections = 1

            is_dummy_section = True
            dummy_frame_count = ((self.dummy_latents - 1) * 4) + 1

        if self.verbose_messaging:
            print("latent_paddings: {}".format(latent_paddings))
            print("noise batches length: {}".format(len(noise)))

        generated_latents = None
        interrupted = False
        real_history_latents = None
        context_latents = None
        generation_status = ""
        temp_history_latents = None
        dummy_images = None
        dummy_gen_latents = None
        generated_image_batches = []
        valid_output = False

        self.cleanup(unload_models=True, cleanup_models=False, cleanup_cuda=True)

        try:
            for i, latent_padding in enumerate(latent_paddings):
                if not self.gen_dummy_only:
                    is_first_section = latent_padding == min(latent_paddings)
                    is_last_section = latent_padding == max(latent_paddings)
                else:
                    is_first_section = True
                    is_last_section = True

                if self.verbose_messaging:
                    print("history_latents Shape: {}".format(history_latents.shape))

                if not is_dummy_section:
                    noise_latent = noise[latent_padding]
                else:
                    noise_latent = dummy_noise

                latent_padding_size = latent_padding * self.latent_window_size

                if self.verbose_messaging:
                    print("latent_padding_size {}: {}  |  latent_padding: {} latent_window_size: {}".format(latent_padding, latent_padding_size, latent_padding, self.latent_window_size))

                if self.embed_interpolation != "disabled":
                    if self.embed_interpolation == "linear":
                        if total_latent_sections <= 1:
                            frac = 1.0  # Handle case with only one section
                        else:
                            frac = 1 - (latent_padding / (total_latent_sections - 1))  # going backwards
                    else:
                        frac = self.start_embed_strength if has_end_image else 1.0

                    image_encoder_last_hidden_state = ((start_image_encoder_last_hidden_state * frac) + ((1 - frac) * end_image_encoder_last_hidden_state)) * self.start_embed_strength
                else:
                    if is_dummy_section or (is_first_section and not self.gen_dummy):
                        image_encoder_last_hidden_state = start_image_encoder_last_hidden_state * self.start_embed_strength
                    else:
                        image_encoder_last_hidden_state = start_image_encoder_last_hidden_state * self.secondary_embed_strength

                start_latent_frames = T  # 0 or 1

                if self.verbose_messaging:
                    print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}, is_dummy_section = {is_dummy_section}')

                if not is_dummy_section:
                    indices = torch.arange(0, sum([1, 16, 2, 1, self.latent_window_size])).unsqueeze(0)
                    clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, self.latent_window_size], dim=1)
                    clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)
                    clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
                else:
                    indices = torch.arange(0, sum([1, 16, 2, 1, self.dummy_latents])).unsqueeze(0)
                    clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, self.dummy_latents], dim=1)
                    clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)
                    clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)

                if is_last_section and (not end_latent is None):
                    clean_latents = torch.cat([start_latent.to(history_latents), end_latent.to(history_latents)], dim=2)
                else:
                    clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

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
                    print("noise Shape: {}\n".format(noise_latent.shape))

                if (self.use_teacache and not is_dummy_section) or (is_dummy_section and self.dummy_teacache):
                    self.transformer.__class__.forward = self.transformer.orig_forward
                    self.transformer.initialize_teacache(enable_teacache=True, num_steps=self.steps, rel_l1_thresh=self.teacache_rel_l1_thresh, verbose_messaging=self.verbose_messaging)
                    self.transformer.initialize_magcache(enable_magcache=False)

                    if self.verbose_messaging:
                        print("TeaCache Enabled")
                elif (self.use_magcache and not is_dummy_section) or (is_dummy_section and self.dummy_magcache):
                    self.transformer.__class__.forward = self.transformer.__class__.magcache_framepack_forward
                    self.transformer.initialize_magcache(enable_magcache=True, num_steps=self.steps, magcache_thresh=self.magcache_thresh, K=self.magcache_K, retention_ratio=self.magcache_retention_ratio, verbose_messaging=self.verbose_messaging)
                    self.transformer.initialize_teacache(enable_teacache=False)

                    if self.verbose_messaging:
                        print("MagCache Enabled")
                else:
                    self.transformer.__class__.forward = self.transformer.orig_forward
                    self.transformer.initialize_teacache(enable_teacache=False)
                    self.transformer.initialize_magcache(enable_magcache=False)

                    if self.verbose_messaging:
                        print("Both TeaCache and MagCache Disabled")

                if not is_dummy_section:
                    print("Generating Batch {}".format(latent_padding))
                else:
                    print("Generating Dummy Batch.")

                generated_latents = self.generate_video(W * 8, H * 8, self.num_frames if not is_dummy_section else dummy_frame_count, rnd, noise_latent, self.llama_vec, self.llama_attention_mask, self.clip_l_pooler, self.llama_vec_n,
                                                        self.llama_attention_mask_n, self.clip_l_pooler_n, image_encoder_last_hidden_state, latent_indices, clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices,
                                                        clean_latents_4x, clean_latent_4x_indices, callback)
                if not is_dummy_section:
                    noise[latent_padding] = None
                else:
                    dummy_noise = None

                if self.verbose_messaging:
                    print("generated_latents {} | Shape: {}".format(latent_padding, generated_latents.shape))

                if not is_dummy_section:
                    offload_model_from_device_for_memory_preservation(self.transformer, self.device, preserved_memory_gb=0)
                    self.cleanup(unload_models=True, cleanup_models=False, cleanup_cuda=True)

                    decoded_images = partial_decode_tiled(self.vae, generated_latents.clone().detach()  / vae_scaling_factor, self.dec_tile_size, self.dec_overlap, self.dec_temporal_size, self.dec_temporal_overlap)
                    generated_image_batches.append(decoded_images.clone().detach())
                    valid_output = True

                    decoded_tuple = torch.split(decoded_images, 1, dim=0)
                    decoded_split = [item for item in decoded_tuple]
                    decoded_images = decoded_images.to(get_offload_device())
                    decoded_images = None

                    start_image = decoded_split[len(decoded_split) - 1]
                    image_embeds = self.clip_vision_encode(start_image)
                    start_latent = self.encode_batched(start_image, per_batch=self.latent_window_size)

                    start_latent = start_latent["samples"] * vae_scaling_factor

                    if self.verbose_messaging:
                        print("start_latent Shape: {}".format(start_latent.shape))

                    start_image_encoder_last_hidden_state = image_embeds["last_hidden_state"].to(self.base_dtype).to(self.device)
                    history_latents = torch.cat([start_latent.to(original_history_latents), original_history_latents], dim=2)

                    decoded_tuple = None
                    decoded_split = None

                    self.cleanup(unload_models=True, cleanup_models=False, cleanup_cuda=True)

                    if not is_last_section:
                        history_latents = torch.cat([original_history_latents, generated_latents.to(original_history_latents)], dim=2)
                        move_model_to_device_with_memory_preservation(self.transformer, target_device=self.device, preserved_memory_gb=self.gpu_memory_preservation)

                    # latent_batches_gend.append(generated_latents.clone().detach()  / vae_scaling_factor)

                    if self.verbose_messaging:
                        print("history_latents {} | Shape: {}".format(latent_padding, history_latents.shape))

                    total_generated_latent_frames += int(generated_latents.shape[2])
                else:
                    offload_model_from_device_for_memory_preservation(self.transformer, self.device, preserved_memory_gb=0)
                    self.cleanup(unload_models=True, cleanup_models=False, cleanup_cuda=True)

                    dummy_gen_latents = generated_latents.clone().detach()  / vae_scaling_factor
                    dummy_decoded = partial_decode_tiled(self.vae, dummy_gen_latents, self.dec_tile_size, self.dec_overlap, self.dec_temporal_size, self.dec_temporal_overlap)

                    if len(dummy_decoded) < 4:
                        dummy_decoded = dummy_decoded.unsqueeze(0)

                    dummy_images = dummy_decoded.clone().detach()

                    if (self.use_dummy_image == "all") or self.gen_dummy_only:
                        generated_image_batches.append(dummy_images)
                        valid_output = True

                    if not self.gen_dummy_only:
                        dummy_tuple = torch.split(dummy_decoded, 1, dim=0)
                        dummy_split = [item for item in dummy_tuple]
                        dummy_decoded = dummy_decoded.to(get_offload_device())
                        dummy_decoded = None

                        if (self.use_dummy_image == "last") or (self.use_dummy_image == "all"):
                            start_image = dummy_split[len(dummy_split) - 1]
                        elif self.use_dummy_image == "first":
                            start_image = dummy_split[0]
                        elif self.use_dummy_image == "middle":
                            start_image = dummy_split[int(math.ceil(len(dummy_split) / 2))]
                        else:
                            random.seed(self.seed)
                            start_image = dummy_split[random.randrange(0, len(dummy_split) - 1, 1)]

                        image_embeds = self.clip_vision_encode(start_image)
                        start_latent = self.encode_batched(start_image, per_batch=self.latent_window_size)

                        start_latent = start_latent["samples"] * vae_scaling_factor
                        print("start_latent Shape: {}".format(start_latent.shape))

                        start_image_encoder_last_hidden_state = image_embeds["last_hidden_state"].to(self.base_dtype).to(self.device)
                        history_latents = torch.cat([start_latent.to(original_history_latents), original_history_latents], dim=2)

                        is_dummy_section = False
                        dummy_tuple = None
                        dummpy_split = None
                        dummy_decoded = None

                    self.cleanup(unload_models=True, cleanup_models=False, cleanup_cuda=True)

                    if not self.gen_dummy_only:
                        move_model_to_device_with_memory_preservation(self.transformer, target_device=self.device, preserved_memory_gb=self.gpu_memory_preservation)

                generated_latents = None

                if self.verbose_messaging:
                    print("history_latents {} | Shape: {}".format(latent_padding, history_latents.shape))

                if is_last_section:
                    break
        except mm.InterruptProcessingException as ie:
            interrupted = True
            print(f"\nWarpedFramepackSampler: Processing Interrupted.")
            print("WarpedFramepackSampler: Returning only partial results (if any).\n If zero images generated, a blank yellow image will be returned.\n")

            generation_status = f"\nWarpedFramepackSampler: Processing Interrupted."

            pass
        except BaseException as e:
            self.cleanup(unload_models=True, cleanup_models=False, cleanup_cuda=True)

            print(f"\nWarpedFramepackSampler: Exception During Processing: {str(e)}")
            print("WarpedFramepackSampler: Returning only partial results (if any).\n If zero images generated, a blank red image will be returned.\n")
            generation_status = f"WarpedFramepackSampler: Exception During Processing: {str(e)}"
            generation_status = "{}{}".format(generation_status, "WarpedFramepackSampler: Returning only partial results (if any).\nIf zero images generated, a blank red image will be returned.")

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

        # if not self.gen_dummy_only:
        offload_model_from_device_for_memory_preservation(self.transformer, self.device, preserved_memory_gb=0)

        self.cleanup(unload_models=True, cleanup_models=False, cleanup_cuda=True)

        if (len(generated_image_batches) > 0):
            output_images = self.assemble_final_result(generated_image_batches)
        elif not dummy_images is None:
            output_images = dummy_images
        elif not dummy_gen_latents is None:
            output_images = partial_decode_tiled(self.vae, dummy_gen_latents, self.dec_tile_size, self.dec_overlap, self.dec_temporal_size, self.dec_temporal_overlap)
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

        return (output_images, generation_status, self.seed, valid_output,)

    def process_v2v(self, video_image_batch):
        if len(video_image_batch.shape) < 4:
            video_image_batch = video_image_batch.unsqueeze(0)

        self.width = video_image_batch.shape[2]
        self.height = video_image_batch.shape[1]
        self.buffer_length = 19

        image_batch_size = int((video_image_batch.shape[0] - 1) / self.batch_count) + 1
        latent_size_factor = 4
        latent_batch_size = self.decoded_to_encoded_length(image_batch_size)

        rnd = torch.Generator("cpu").manual_seed(self.seed)

        if self.verbose_messaging:
            print("process_v2v: Video Context Images Shape: {}".format(video_image_batch.shape))
            print("process_v2v: image_batch_size: {}".format(image_batch_size))
            print("process_v2v: latent_batch_size: {}".format(latent_batch_size))

        total_latent_sections = self.batch_count

        video_image_batch.to(dtype=torch.float32, device=self.device)
        video_latent_batches, video_encoding_batches = self.video_encode(video_image_batch, image_batch_size)
        video_image_batch.to(dtype=torch.float32, device=self.offload_device)

        B, C, _, H, W = video_latent_batches[0].shape
        T = 1

        output_images = None

        latent_paddings_list = list(range(total_latent_sections))

        total_generated_latent_frames = 0
        latent_paddings = latent_paddings_list.copy()  # Create a copy for iteration

        latent_embeds = self.get_video_latent_embeds(video_image_batch, image_batch_size, latent_paddings)

        comfy_model, patcher, callback = self.initialize_comfy_model()

        print("latent_paddings count: {}  |  latent_paddings: {}".format(len(latent_paddings), latent_paddings))

        # noise = self.setup_latent_noise(W, H)
        noise, _ = self.setup_latent_noise(self.latent_window_size * self.batch_count, W, H)

        if self.verbose_messaging:
            print("Noise Batches Length: {}".format(len(noise)))
            print("sample noise Shape: {}".format(noise[0].shape))

        has_end_image = True
        generated_latents = None
        interrupted = False
        history_latents = None
        real_history_latents = None
        context_latents = None
        generation_status = ""
        temp_history_latents = None
        original_history_latents = None
        valid_output = False
        is_dummy_section = False

        self.cleanup(unload_models=True, cleanup_models=False, cleanup_cuda=True)

        latent_batches_gend = []

        try:
            for padding_i, latent_padding in enumerate(latent_paddings):
                padding_key = "{}".format(latent_padding)
                next_padding_key = "{}".format(latent_padding + 1)

                is_first_section = latent_padding == min(latent_paddings)
                is_last_section = latent_padding == max(latent_paddings)

                start_latent = latent_embeds[padding_key]["start_latent"]
                start_image_encoder_last_hidden_state = latent_embeds[padding_key]["start_embedding"]["last_hidden_state"].to(self.base_dtype).to(self.device)
                image_encoder_last_hidden_state = torch.mul(start_image_encoder_last_hidden_state, self.start_embed_strength)

                if self.verbose_messaging:
                    print("\nlatent_padding: {}  |  latent_window_size: {}".format(latent_padding, self.latent_window_size))

                noise_latent = noise[latent_padding]

                if self.verbose_messaging:
                    print("noise_latent Shape: {}".format(noise_latent.shape))

                if is_first_section:
                    original_history_latents = torch.zeros(size=(1, 16, self.buffer_length, H, W), dtype=torch.float32).cpu()

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

                if (self.use_teacache and not is_dummy_section) or (is_dummy_section and self.dummy_teacache):
                    self.transformer.__class__.forward = self.transformer.orig_forward
                    self.transformer.initialize_teacache(enable_teacache=True, num_steps=self.steps, rel_l1_thresh=self.teacache_rel_l1_thresh, verbose_messaging=self.verbose_messaging)
                    self.transformer.initialize_magcache(enable_magcache=False)

                    if self.verbose_messaging:
                        print("TeaCache Enabled")
                elif (self.use_magcache and not is_dummy_section) or (is_dummy_section and self.dummy_magcache):
                    self.transformer.__class__.forward = self.transformer.__class__.magcache_framepack_forward
                    self.transformer.initialize_magcache(enable_magcache=True, num_steps=self.steps, magcache_thresh=self.magcache_thresh, K=self.magcache_K, retention_ratio=self.magcache_retention_ratio, verbose_messaging=self.verbose_messaging)
                    self.transformer.initialize_teacache(enable_teacache=False)

                    if self.verbose_messaging:
                        print("MagCache Enabled")
                else:
                    self.transformer.__class__.forward = self.transformer.orig_forward
                    self.transformer.initialize_teacache(enable_teacache=False)
                    self.transformer.initialize_magcache(enable_magcache=False)

                    if self.verbose_messaging:
                        print("Both TeaCache and MagCache Disabled")

                clean_latents.to(dtype=torch.float32, device=self.device)
                clean_latents_2x.to(dtype=torch.float32, device=self.device)
                clean_latents_4x.to(dtype=torch.float32, device=self.device)
                noise_latent.to(dtype=torch.float32, device=self.device)

                print("Generating Batch {}".format(latent_padding))

                generated_latents = self.generate_video(W * 8, H * 8, self.num_frames, rnd, noise_latent, self.llama_vec, self.llama_attention_mask, self.clip_l_pooler, self.llama_vec_n, self.llama_attention_mask_n,
                                                        self.clip_l_pooler_n, image_encoder_last_hidden_state, latent_indices, clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices,
                                                        clean_latents_4x, clean_latent_4x_indices, callback)
                noise[latent_padding] = None

                if self.verbose_messaging:
                    print("generated_latents for batch {}: Shape: {}\n".format(latent_padding, generated_latents.shape))

                latent_batches_gend.append(generated_latents.clone().detach() / vae_scaling_factor)
                valid_output = True
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
            print(f"\nWarpedFramepackSampler: Processing Interrupted.")
            print("WarpedFramepackSampler: Returning only partial results (if any).\n If zero images generated, a blank yellow image will be returned.\n")

            generation_status = f"\nWarpedFramepackSampler: Processing Interrupted."

            pass
        except BaseException as e:
            self.cleanup(unload_models=True, cleanup_models=False, cleanup_cuda=True)

            print(f"\nWarpedFramepackSampler: Exception During Processing: {str(e)}")
            print("WarpedFramepackSampler: Returning only partial results (if any).\n If zero images generated, a blank red image will be returned.\n")
            generation_status = f"WarpedFramepackSampler: Exception During Processing: {str(e)}"
            generation_status = "{}{}".format(generation_status, "WarpedFramepackSampler: Returning only partial results (if any).\nIf zero images generated, a blank red image will be returned.")

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

        # self.transformer.to(self.offload_device)
        offload_model_from_device_for_memory_preservation(self.transformer, self.device, preserved_memory_gb=0)

        self.cleanup(unload_models=True, cleanup_models=False, cleanup_cuda=True)

        if len(latent_batches_gend) > 0:
            output_images = self.decode_batches(latent_batches_gend, self.skip_frames)
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

        return (output_images, generation_status, self.seed, valid_output,)

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

    def setup_latent_noise(self, noise_latent_length, width, height):
        rnd = torch.Generator("cpu").manual_seed(self.seed)
        dummy_noise = None

        if not self.gen_dummy or (self.mode == VideoGenerationType.V2V):
            temp_noise = torch.randn((1, 16, noise_latent_length, height, width), generator=rnd, device=rnd.device).to(device=rnd.device, dtype=torch.float32)
        else:
            dummy_noise = torch.randn((1, 16, self.dummy_latents, height, width), generator=rnd, device=rnd.device).to(device=rnd.device, dtype=torch.float32)

            rnd = torch.Generator("cpu").manual_seed(self.seed)
            temp_noise = torch.randn((1, 16, noise_latent_length + self.dummy_latents - 1, height, width), generator=rnd, device=rnd.device).to(device=rnd.device, dtype=torch.float32)

            temp_noise_tuple = torch.split(temp_noise, self.dummy_latents, dim=2)
            temp_noise_split = [item for item in temp_noise_tuple]
            temp_noise_tuple = None

            # dummy_noise = temp_noise_split[0]

            dummy_noise_tuple = torch.split(dummy_noise, 1, dim=2)
            dummy_noise_split = [item for item in dummy_noise_tuple]
            dummy_noise_tuple = None

            temp_noise = dummy_noise_split[len(dummy_noise_split) - 1]
            dummy_noise_split = None

            i = -1
            for entry in temp_noise_split:
                i += 1

                if i > 0:
                    if not temp_noise is None:
                        temp_noise = torch.cat((temp_noise, entry), 2)
                    else:
                        temp_noise = entry

            temp_noise_split = None

            gc.collect()
            time.sleep(1)

        if Decimal(self.noise_strength).compare(Decimal(1.00)) != 0:
            temp_noise = torch.mul(temp_noise, self.noise_strength)

        temp_noise_tuple = torch.split(temp_noise, self.latent_window_size, dim=2)
        noise = [item for item in temp_noise_tuple]
        temp_noise = None
        temp_noise_tuple = None

        if self.verbose_messaging:
            if not dummy_noise is None:
                print("\nsetup_latent_noise: dummy_noise Shape: {}".format(dummy_noise.shape))

            i = 0
            for entry in noise:
                print("setup_latent_noise: Batch: {}  |  noise Shape: {}".format(i, entry.shape))
                i += 1

            print("\n")

        return noise, dummy_noise

    def initialize_comfy_model(self):
        comfy_model = HyVideoModel(
                HyVideoModelConfig(self.base_dtype),
                model_type=comfy.model_base.ModelType.FLOW,
                device=self.device,
            )

        patcher = comfy.model_patcher.ModelPatcher(comfy_model, self.device, torch.device("cpu"))
        callback = latent_preview.prepare_callback(patcher, self.steps)

        move_model_to_device_with_memory_preservation(self.transformer, target_device=self.device, preserved_memory_gb=self.gpu_memory_preservation)

        return comfy_model, patcher, callback

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
        elif context_frames > self.latent_window_size:
            context_frames = self.latent_window_size

        if context_frames >= video_latents.shape[2]:
            clean_latent_indices = []
            i = 0
            while i < video_latents.shape[2]:
                clean_latent_indices.append(i)

                i +=1
        elif context_frames > 3:
            if context_frames > (video_latents.shape[2] - 3):
                context_frames = video_latents.shape[2] - 3

            offset = math.ceil((video_latents.shape[2] - 2 - self.buffer_length) / (context_frames - 2))

            if offset < 1:
                offset = 1

            print("offset: {}".format(offset))
            index = 0
            i = 0
            clean_latent_indices = []
            temp_clean_latent_indices = {}

            while i < (context_frames - 2):
                if index < (video_latents.shape[2] - 1):
                    temp_clean_latent_indices["{}".format(int(index + self.buffer_length))] = int(index + self.buffer_length)

                index += offset
                i += 1

            temp_clean_latent_indices["{}".format(int(video_latents.shape[2] - 2))] = int(video_latents.shape[2] - 2)
            temp_clean_latent_indices["{}".format(int(video_latents.shape[2] - 1))] = int(video_latents.shape[2] - 1)

            if self.verbose_messaging:
                print("temp_clean_latent_indices.items(): {}".format(temp_clean_latent_indices.items()))

            for key in temp_clean_latent_indices:
                clean_latent_indices.append(temp_clean_latent_indices[key])

            clean_latent_indices.sort()

            clean_latents = None
            for index in clean_latent_indices:
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

    def setup_dummy_noise(self, width, height):
        rnd = torch.Generator("cpu").manual_seed(self.seed)
        noise = torch.randn((1, 16, self.dummy_latents, height, width), generator=rnd, device=rnd.device).to(device=rnd.device, dtype=torch.float32)

        if Decimal(self.noise_strength).compare(Decimal(1.00)) != 0:
            noise = torch.mul(temp_noise, self.noise_strength)

        return noise

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

        if self.verbose_messaging:
            print("get_latent_window_size: Next Lowest Initialized To: {}".format(next_lowest))
            print("get_latent_window_size: Next Highest Initialized To: {}".format(next_highest))

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

        print("Unable to find a compatible latent_window_size for number of frames = {} and preferred_batch_size = {}.".format(frame_count, preferred_batch_size))
        print("Recalculating Number Of Frames Based On preferred_batch_size of: {}".format(preferred_batch_size))

        return self.calculate_new_number_of_frames(preferred_batch_size, (((frame_count - 1) // 4) * 4) + 1, use_batch_size)

    def calculate_new_number_of_frames(self, preferred_batch_size, frame_count, use_batch_size):
        working_batch_size = preferred_batch_size - 1
        working_frame_count = frame_count - 1

        next_lowest = next_highest = working_frame_count
        next_lowest_found = False
        next_highest_found = False

        while next_lowest > 37:
            next_lowest -= 4

            if int(next_lowest % working_batch_size) == 0:
                next_lowest_found = True
                break

        if next_lowest_found and (use_batch_size == "next_lowest"):
            return self.decoded_to_encoded_length(preferred_batch_size), int(next_lowest // working_batch_size), next_lowest + 1

        while next_highest < 999997:
            next_highest += 4

            if int(next_highest % working_batch_size) == 0:
                next_highest_found = True
                break

        if next_highest_found and (use_batch_size == "next_highest"):
            return self.decoded_to_encoded_length(preferred_batch_size), int(next_highest // working_batch_size), next_highest + 1

        if next_lowest_found and next_highest_found:
            if (working_frame_count - next_lowest) <= (next_highest - working_frame_count):
                return self.decoded_to_encoded_length(preferred_batch_size), int(next_lowest // working_batch_size), next_lowest + 1

            return self.decoded_to_encoded_length(preferred_batch_size), int(next_highest // working_batch_size), next_highest + 1

        if next_lowest_found:
            return self.decoded_to_encoded_length(preferred_batch_size), int(next_lowest // working_batch_size), next_lowest + 1

        if next_highest_found:
            return self.decoded_to_encoded_length(preferred_batch_size), int(next_highest // working_batch_size), next_highest + 1

        raise ValueError("Unable to find a compatible latent_window_size for number of frames = {} and preferred_batch_size = {}.".format(frame_count, preferred_batch_size))

    def clip_vision_encode(self, image, crop="center"):
            crop_image = True
            if crop != "center":
                crop_image = False
            output = self.clip_vision.encode_image(image, crop=crop_image)
            return output

    def get_video_latent_embeds(self, video_frames, batch_frame_count, paddings):
        if self.verbose_messaging:
            print("get_video_latent_embeds: video_frames Shape: {}".format(video_frames.shape))
            print("get_video_latent_embeds: batch_frame_count: {}".format(batch_frame_count))
            print("get_video_latent_embeds: paddings: {}".format(paddings))

        if len(video_frames.shape) < 4:
            video_frames = video_frames.unsqueeze(0)

        first_batch = video_frames[:batch_frame_count, :, :, :]
        remaining_batches = video_frames[batch_frame_count:, :, :, :]

        start_image = first_batch[1:2, :, :, :]
        end_image = first_batch[(batch_frame_count - 1):, :, :, :]
        next_start_image = first_batch[(first_batch.shape[0] - 4):(first_batch.shape[0] - 3), :, :, :]

        start_latent = self.encode_batched(start_image, self.latent_window_size)
        start_latent_embedding = self.clip_vision_encode(start_image)
        start_image = start_image.to(self.offload_device)
        end_latent = self.encode_batched(end_image, self.latent_window_size)
        end_latent_embedding = self.clip_vision_encode(end_image)
        end_image = end_image.to(self.offload_device)
        next_first_latent = end_latent["samples"].clone().detach().to(end_latent["samples"])
        next_first_embedding = copy.deepcopy(end_latent_embedding)

        if self.verbose_messaging:
            print("get_video_latent_embeds: first_batch Shape: {}".format(first_batch.shape))
            print("get_video_latent_embeds: remaining_batches Shape: {}".format(remaining_batches.shape))
            print("get_video_latent_embeds: batch 0: start_image Shape: {}".format(start_image.shape))
            print("get_video_latent_embeds: batch 0: start_latent Shape: {}".format(start_latent["samples"].shape))
            print("get_video_latent_embeds: batch 0: end_image Shape: {}".format(end_image.shape))
            print("get_video_latent_embeds: batch 0: end_latent Shape: {}".format(end_latent["samples"].shape))
            print("get_video_latent_embeds: batch 0: next_start_image Shape: {}".format(next_start_image.shape))
            print("get_video_latent_embeds: batch 0: next_first_latent Shape: {}".format(next_first_latent.shape))

        latent_embeds = {"0": {}}
        latent_embeds["0"]["start_latent"] = (start_latent["samples"] * vae_scaling_factor).to(self.offload_device)
        latent_embeds["0"]["start_embedding"] = start_latent_embedding
        latent_embeds["0"]["end_latent"] = (end_latent["samples"] * vae_scaling_factor).to(self.offload_device)
        latent_embeds["0"]["end_embedding"] = end_latent_embedding

        start_image = None
        start_latent = None
        start_latent_embedding = None
        end_image = None
        end_latent = None
        end_latent_embedding = None
        next_start_four = None
        next_start_image = None

        if remaining_batches.shape[0] > 0:
            image_batches_tuple = torch.split(remaining_batches, batch_frame_count - 1, dim=0)
            image_batches_split = [item for item in image_batches_tuple]
            image_batches_tuple = None

            if self.verbose_messaging:
                i = 0
                for entry in image_batches_split:
                    print("get_video_latent_embeds: batch: {}  |  split Shape: {}".format(i + 1, image_batches_split[i].shape))
                    i += 1

            print("get_video_latent_embeds Batch: Encoding Start/End Images...")

            batch_number = 1
            for batch in image_batches_split:
                batch_key = "{}".format(batch_number)

                if self.verbose_messaging:
                    print("get_video_latent_embeds: Processing Batch: {}".format(batch_number))

                latent_embeds[batch_key] = {"start_latent": None, "end_latent": None, "start_embedding": None, "end_embedding": None}

                start_latent = next_first_latent
                start_latent_embedding = next_first_embedding

                end_image = batch[(batch.shape[0] - 1):, :, :, :]
                end_image = end_image.to(self.device)
                end_latent = self.encode_batched(end_image, self.latent_window_size)
                # end_latent = end_latent["samples"]
                end_latent_embedding = self.clip_vision_encode(end_image)
                end_image = end_image.to(self.device)

                latent_embeds[batch_key]["start_latent"] = (start_latent * vae_scaling_factor).to(device=self.offload_device)
                latent_embeds[batch_key]["start_embedding"] = start_latent_embedding
                latent_embeds[batch_key]["end_latent"] = (end_latent["samples"] * vae_scaling_factor).to(device=self.offload_device)
                latent_embeds[batch_key]["end_embedding"] = end_latent_embedding

                if batch_number < max(paddings):
                    next_start_image = batch[(batch.shape[0] - 1):, :, :, :]
                    next_first_latent = self.encode_batched(next_start_image, self.latent_window_size)
                    next_first_latent = next_first_latent["samples"]
                    next_first_embedding = copy.deepcopy(end_latent_embedding)
                    next_start_image = None
                else:
                    next_first_embedding = None
                    next_start_image = None
                    next_first_latent = None

                start_image = None
                start_latent = None
                start_latent_embedding = None
                end_image = None
                end_latent = None
                end_latent_embedding = None
                batch_image_batches_split = None

                batch_number += 1

            print("get_video_latent_embeds Batch: Encoding Start/End Images...Done")

            if self.verbose_messaging:
                print("get_video_latent_embeds: latent_embeds length: {}".format(len(latent_embeds)))

                for batch_key in latent_embeds.keys():
                    print("get_video_latent_embeds: Batch {}: start_latent Shape: {}".format(batch_key, latent_embeds[batch_key]["start_latent"].shape))
                    print("get_video_latent_embeds: Batch {}: end_latent Shape: {}".format(batch_key, latent_embeds[batch_key]["end_latent"].shape))
                    print("get_video_latent_embeds: Batch {}: start_embedding Shape: {}".format(batch_key, latent_embeds[batch_key]["start_embedding"]["last_hidden_state"].shape))
                    print("get_video_latent_embeds: Batch {}: end_embedding Shape: {}".format(batch_key, latent_embeds[batch_key]["end_embedding"]["last_hidden_state"].shape))

        self.cleanup(unload_models=False, cleanup_models=False, cleanup_cuda=True)

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

    def decode_batches(self, latent_batches, skip_frames):
        if (latent_batches is None) or (len(latent_batches) < 1):
            print("decode_batches: Warning...nothing to decode.")
            return None

        self.cleanup(unload_models=True, cleanup_models=False, cleanup_cuda=False)

        resulting_images = None

        if self.blend_frames < 1:
            i = 0
            for entry in latent_batches:
                entry.to(dtype=torch.bfloat16, device=self.device)

                if i < (len(latent_batches) - 1):
                    temp_decoded = self.decode_tiled(entry, unload_after=False)
                else:
                    temp_decoded = self.decode_tiled(entry, unload_after=True)

                if len(temp_decoded.shape) < 4:
                    temp_decoded = temp_decoded.unsqueeze(0)

                # if i > 0:
                #     temp_decoded = temp_decoded[3:, :, :, :]

                if not resulting_images is None:
                    resulting_images = torch.cat((resulting_images, temp_decoded), 0)
                else:
                    resulting_images = temp_decoded

                entry.to(device=self.offload_device)

                self.cleanup(unload_models=False, cleanup_models=False, cleanup_cuda=True)

                i += 1
        else:
            temp_decoded_batches = []

            i = 0
            for entry in latent_batches:
                entry.to(dtype=torch.bfloat16, device=self.device)

                if i < (len(latent_batches) - 1):
                    temp_decoded = self.decode_tiled(entry, unload_after=False)
                else:
                    temp_decoded = self.decode_tiled(entry, unload_after=True)

                if len(temp_decoded.shape) < 4:
                    temp_decoded = temp_decoded.unsqueeze(0)

                temp_decoded_batches.append(temp_decoded)

                entry.to(device=self.offload_device)

                self.cleanup(unload_models=False, cleanup_models=False, cleanup_cuda=True)

                i += 1

            resulting_images = self.assemble_final_result(temp_decoded_batches)
            temp_decoded_batches = None
            self.cleanup(unload_models=False, cleanup_models=False, cleanup_cuda=True)

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

    def assemble_final_result(self, image_batches):
        if self.blend_frames < 1:
            resulting_images = None
            for entry in image_batches:
                if not resulting_images is None:
                    resulting_images = torch.cat((resulting_images, entry), 0)
                else:
                    resulting_images = entry

                entry.to(device=self.offload_device)
                entry = None
        else:
            blend_value = 1.0 / self.blend_frames
            i = 0
            while i < (len(image_batches) - 1):
                alpha_blend_val = blend_value
                blend_count = self.blend_frames

                image_batches_tuple = torch.split(image_batches[i], 1, dim=0)
                image_batches_split = [item for item in image_batches_tuple]
                image1 = image_batches_split[len(image_batches_split) - 1]
                # image1 = image_batches_split[len(image_batches_split) - 4]
                image_batches_tuple = None
                image_batches_split = None

                image_batches_tuple = torch.split(image_batches[i + 1], 1, dim=0)
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

                image_batches_tuple = torch.split(image_batches[i], image_batches[i].shape[0] - 3, dim=0)
                image_batches_split = [item for item in image_batches_tuple]
                image_batches_tuple = None

                image_batches[i] = torch.cat((image_batches_split[0], blend_latents), 0)
                blend_latents = None
                image_batches_split = None

                self.cleanup(unload_models=False, cleanup_models=False, cleanup_cuda=True)

                i += 1

            resulting_images = None
            for entry in image_batches:
                if not resulting_images is None:
                    resulting_images = torch.cat((resulting_images, entry), 0)
                else:
                    resulting_images = entry

                entry.to(device=self.offload_device)

            image_batches = None

            self.cleanup(unload_models=False, cleanup_models=False, cleanup_cuda=True)


        print("assemble_final_result: Full decoded images count: {}".format(resulting_images.shape[0]))

        if self.skip_frames < 1:
            return resulting_images

        skipped_frames = 1

        image_batches_tuple = torch.split(resulting_images, 1, dim=0)
        image_batches_split = [item for item in image_batches_tuple]

        resulting_images = None

        for image in image_batches_split:
            if skipped_frames <= self.skip_frames:
                skipped_frames += 1
                continue

            if not resulting_images is None:
                resulting_images = torch.cat((resulting_images, image), 0)
            else:
                resulting_images = image

        print("assemble_final_result: Final decoded images count: {}".format(resulting_images.shape[0]))

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

    def cleanup(self, unload_models=False, cleanup_models=False, cleanup_cuda=False):
        if unload_models:
            mm.unload_all_models()

        if cleanup_models:
            mm.cleanup_models()

        if cleanup_cuda:
            mm.soft_empty_cache()

        gc.collect()
        time.sleep(1)

        return

    def video_encode(self, video_frames, batch_size):
        if len(video_frames.shape) < 4:
            video_frames = video_frames.unsqueeze(0)

        print(f"video_encode: Encoding input video frames in batch size {batch_size} (reduce preferred_batch_size if memory issues here or if forcing video resolution)")

        if self.verbose_messaging:
            print("video_encode: video_frames Shape: {}".format(video_frames.shape))
            print("video_encode: batch_size: {}".format(batch_size))

        first_batch = video_frames[:batch_size, :, :, :]
        remaining_batches = video_frames[batch_size:, :, :, :]
        # next_start_image = first_batch[(first_batch.shape[0] - 4):(first_batch.shape[0] - 3), :, :, :]
        next_start_four = first_batch[(first_batch.shape[0] - 4):, :, :, :]

        first_batch_latents = partial_encode_tiled(self.vae, first_batch, unload_after=False)

        if self.verbose_messaging:
            print("video_encode: first_batch Shape: {}".format(first_batch.shape))
            print("video_encode: remaining_batches Shape: {}".format(remaining_batches.shape))
            # print("video_encode: next_start_image Shape: {}".format(next_start_image.shape))
            print("video_encode: next_start_four Shape: {}".format(next_start_four.shape))

        if len(first_batch_latents.shape) < 5:
            first_batch_latents = first_batch_latents.unsqueeze(0)

        first_batch_latents = first_batch_latents * vae_scaling_factor
        final_latent_batches = [first_batch_latents.to(self.offload_device)]

        image_batches_tuple = torch.split(remaining_batches, batch_size - 1, dim=0)
        image_batches_split = [item for item in image_batches_tuple]

        final_image_batches = []
        last_frame = None

        latents = None
        batch_encodings = []

        with torch.no_grad():
            i = 0
            for entry in image_batches_split:
                entry = entry.to(dtype=torch.bfloat16, device=self.device)
                entry = torch.cat((next_start_four.to(entry), entry), 0)
                # next_start_image = entry[(entry.shape[0] - 1):, :, :, :]
                next_start_four = entry[(entry.shape[0] - 4):, :, :, :]

                if i < (len(image_batches_split) - 1):
                    batch_latent = partial_encode_tiled(self.vae, entry, unload_after=False)
                else:
                    batch_latent = partial_encode_tiled(self.vae, entry, unload_after=True)

                if len(batch_latent.shape) < 5:
                    batch_latent = batch_latent.unsqueeze(0)

                batch_latent = batch_latent * vae_scaling_factor
                batch_latent = batch_latent.to(self.offload_device)
                final_latent_batches.append(batch_latent.clone().detach())
                entry = entry.to(device=self.offload_device)
                entry = None
                batch_latent = None

        print(f"Encoding input video frames in batch size {batch_size} Done.")

        if self.verbose_messaging:
            i = 0
            for entry in final_latent_batches:
                print("video_encode: Batch: {}  |  batch Shape: {}".format(i, entry.shape))
                i += 1

        return final_latent_batches, batch_encodings

def cleanup(device=None, unload_models=False, cleanup_cuda=False):
    if unload_models:
        mm.free_memory(1e30, device)

    if cleanup_cuda:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()
    time.sleep(1)

    return

class WarpedFramepackLoraSelectBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora": ("STRING", {"default": "", "forceInput": True}),
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

def warped_load_torch_file(ckpt, return_metadata=False):
    from safetensors.torch import load as safeload

    metadata = {}

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

def warped_load_lora_weights(lora_name, return_metadata=False):
    if (lora_name is None) or (len(lora_name) < 1):
        raise ValueError("lora_name cannot be None or Empty.")

    # Get the full path to the LoRA file
    lora_path = folder_paths.get_full_path("loras", lora_name)

    if not os.path.exists(lora_path):
        raise ValueError(f"LoRA file not found: {lora_path}")

    print("Reading LORA At Path: {}".format(lora_path))
    with open(lora_path, "rb") as file:
        lora_weights = file.read()

    print("Loading Lora: {}...".format(lora_name))
    lora_sd, metadata = warped_load_torch_file(lora_weights, return_metadata=True)
    print("Loading Lora: {}...Done".format(lora_name))

    if return_metadata:
        return lora_sd, metadata

    return lora_sd

def get_available_devices():
    available_devices = ["cpu"]

    if torch.cuda.is_available():
        available_devices.append("cuda")

        if torch.cuda.device_count() > 1:
            for i in range(torch.cuda.device_count()):
                temp_device = "cuda:{}".format(i)
                available_devices.append(temp_device)

    return available_devices

class WarpedDualEncoder(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(self):
        return {"required": { "clip": ("CLIP", ),
                              "positive_text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The positive prompt to be encoded."}),
                              "negative_text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The negative prompt to be encoded."}),
                            },
               }
    RETURN_TYPES = (IO.CONDITIONING, IO.CONDITIONING, "STRING", "STRING", )
    RETURN_NAMES = ("pos_conditioning", "neg_conditioning", "pos_prompt", "neg_prompt", )
    FUNCTION = "process"
    CATEGORY = "Warped/General/Conditioning"
    DESCRIPTION = "Encodes both positive and negative prompts."

    def process(self, clip, positive_text="", negative_text=""):
        print("WarpedDualEncoder: Loading clip model to device: {}".format(clip.patcher.load_device))
        clip.patcher.model.to(device=clip.patcher.load_device)

        print("WarpedDualEncoder: Encoding Prompts...")
        positive_conditioning = self.encode(clip, positive_text)
        negative_conditioning = self.encode(clip, negative_text)
        print("WarpedDualEncoder: Encoding Prompts...Done.")

        print("WarpedDualEncoder: Unloading clip model to device: {}".format(get_offload_device()))
        clip.patcher.model.to(device=get_offload_device())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        time.sleep(1)

        return (positive_conditioning, negative_conditioning, positive_text, negative_text, )

    def encode(self, clip, text):

        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        tokens = clip.tokenize(text)
        return_encoding = clip.encode_from_tokens_scheduled(tokens)

        return return_encoding

class WarpedSingleEncoder(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(self):
        return {"required": { "clip": ("CLIP", ),
                              "input_text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The prompt to be encoded."}),
                            },
               }
    RETURN_TYPES = (IO.CONDITIONING, "STRING", )
    RETURN_NAMES = ("conditioning", "prompt")
    FUNCTION = "process"
    CATEGORY = "Warped/General/Conditioning"
    DESCRIPTION = "Encodes a single prompt."

    def process(self, clip, input_text=""):
        print("WarpedSingleEncoder: Loading clip model to device: {}".format(clip.patcher.load_device))
        clip.patcher.model.to(device=clip.patcher.load_device)

        print("WarpedSingleEncoder: Encoding Prompt...")
        output_conditioning = self.encode(clip, input_text)
        print("WarpedSingleEncoder: Encoding Prompt...Done.")

        print("WarpedSingleEncoder: Unloading clip model to device: {}".format(get_offload_device()))
        clip.patcher.model.to(device=get_offload_device())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        time.sleep(1)

        return (output_conditioning, input_text, )

    def encode(self, clip, text):

        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        tokens = clip.tokenize(text)
        return_encoding = clip.encode_from_tokens_scheduled(tokens)

        return return_encoding

class WarpedCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
                              "clip_type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace"], ),
                              },
                "optional": {
                              "device": (get_available_devices(), {"default": "cpu"}),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "Warped/General/Loaders"

    DESCRIPTION = "[Recipes]\n\nstable_diffusion: clip-l\nstable_cascade: clip-g\nsd3: t5 xxl/ clip-g / clip-l\nstable_audio: t5 base\nmochi: t5 xxl\ncosmos: old t5 xxl\nlumina2: gemma 2 2B\nwan: umt5 xxl\n hidream: llama-3.1 (Recommend) or t5"

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

    def encode(self, clip, text):

        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        tokens = clip.tokenize(text)
        return_encoding = clip.encode_from_tokens_scheduled(tokens)

        return return_encoding

class WarpedDualCLIPLoader:
    @classmethod
    def INPUT_TYPES(self):
        return {"required": { "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                              "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
                              "type": (["sdxl", "sd3", "flux", "hunyuan_video", "hunyuan_video_15"], ),
                              "device": (get_available_devices(), {"default": "cuda"}),
                            },
               }
    RETURN_TYPES = ("CLIP", )
    RETURN_NAMES = ("clip", )
    FUNCTION = "load_clip"

    CATEGORY = "Warped/General/Loaders"

    DESCRIPTION = "[Recipes]\n\nsdxl: clip-l, clip-g\nsd3: clip-l, clip-g / clip-l, t5 / clip-g, t5\nflux: clip-l, t5"

    def load_clip(self, clip_name1, clip_name2, type, device="cpu"):
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        if type == "sdxl":
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "flux":
            clip_type = comfy.sd.CLIPType.FLUX
        elif type == "hunyuan_video":
            clip_type = comfy.sd.CLIPType.HUNYUAN_VIDEO
        elif type == "hunyuan_video_15":
            clip_type = comfy.sd.CLIPType.HUNYUAN_VIDEO_15

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

        print("WarpedDualCLIPLoader: {}".format(model_options))

        clip = self.sd_load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)

        return (clip, )

    def sd_load_clip(self, ckpt_paths, embedding_directory=None, clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options={"offload_device": get_offload_device()}):
        checkpoint_temp = []
        for p in ckpt_paths:
            print("Reading: {}".format(p))
            with open(p, "rb") as file:
                content = file.read()

            checkpoint_temp.append(content)

        clip_data = []
        i = 1
        for p in checkpoint_temp:
            print("Loading Clip: {}...".format(i))
            clip_data.append(warped_load_torch_file(p))
            print("Loading Clip: {}...Done".format(i))
            i += 1

        return_clip = comfy.sd.load_text_encoder_state_dicts(clip_data, embedding_directory=embedding_directory, clip_type=clip_type, model_options=model_options)

        checkpoint_temp = None
        clip_data = None

        return return_clip

    def encode(self, clip, text):

        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        tokens = clip.tokenize(text)
        return_encoding = clip.encode_from_tokens_scheduled(tokens)

        return return_encoding

script_directory = os.path.dirname(os.path.abspath(__file__))

class WarpedLoadFramePackModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "load_device": (get_available_devices(), {"default": "cuda", "tooltip": "set load device."}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn",
                    "sageattn",
                    ], {"default": "sdpa"}),
                "compile_args": ("FRAMEPACKCOMPILEARGS", ),
                "lora": ("FPLORA", {"default": None, "tooltip": "LORA model to load"}),
            }
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "Warped/Framepack/Loaders"

    def loadmodel(self, model, base_precision, quantization,
                  compile_args=None, attention_mode="sdpa", lora=None, load_device="main_device"):
        from .diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModel
        from accelerate.utils import set_module_tensor_to_device
        from accelerate import init_empty_weights
        from .diffusers_helper.memory import DynamicSwapInstaller, move_model_to_device_with_memory_preservation

        mm.unload_all_models()
        gc.collect()
        time.sleep(1)

        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]

        offload_device = get_offload_device()

        if load_device == "cuda":
            device = mm.get_torch_device()
        elif load_device == "cpu":
            device = offload_device
        else:
            temp_strings = load_device.split(':')
            device = torch.device(int(temp_strings[len(temp_strings) - 1]))

        transformer_load_device = offload_device

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        model_config_path = os.path.join(script_directory, "transformer_config.json")
        import json
        with open(model_config_path, "r") as f:
            config = json.load(f)

        print("Reading: {}".format(model_path))
        with open(model_path, "rb") as file:
            checkpoint_temp = file.read()

        print("Loading Checkpoint...")
        sd = warped_load_torch_file(checkpoint_temp)
        print("Loading Checkpoint...Done")
        checkpoint_temp = None

        gc.collect()
        time.sleep(1)

        # for key in sd.keys():
        #     print("Framepack Model Key: {}".format(key))

        # sd = load_torch_file(model_path, device=offload_device, safe_load=True)
        model_weight_dtype = sd['single_transformer_blocks.0.attn.to_k.weight'].dtype

        with init_empty_weights():
            transformer = HunyuanVideoTransformer3DModel(**config, attention_mode=attention_mode)

        params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
        if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast" or quantization == "fp8_scaled":
            dtype = torch.float8_e4m3fn
        elif quantization == "fp8_e5m2":
            dtype = torch.float8_e5m2
        else:
            dtype = base_dtype

        if lora is not None:
            after_lora_dtype = dtype
            dtype = base_dtype

        print("Using accelerate to load and assign model weights to device...")
        param_count = sum(1 for _ in transformer.named_parameters())
        for name, param in tqdm(transformer.named_parameters(),
                desc=f"Loading transformer parameters to {transformer_load_device}",
                total=param_count,
                leave=True):
            dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype

            set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=dtype_to_use, value=sd[name])

        if lora is not None:
            from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers

            adapter_list = []
            adapter_weights = []

            lora_number = 1
            for l in lora:
                fuse = True if l["fuse_lora"] else False

                blocktypes = None

                if "blocks" in l.keys():
                    blocktypes = l["blocks"]

                print("Reading LORA: {}".format(l["path"]))
                with open(l["path"], "rb") as file:
                    lora_temp = file.read()

                print("Loading Lora: {}...".format(lora_number))
                lora_sd = warped_load_torch_file(lora_temp)
                print("Loading Lora: {}...Done".format(lora_number))
                lora_number += 1

                # Convert Lora To diffusion_model format
                lora_sd = convert_lora(lora_sd, convert_to="diffusion_model")

                # for key in lora_sd:
                #     print("Lora Key: {}".format(key))

                # If blocks value exist in lora metadata, then select block types
                if not blocktypes is None:
                    lora_sd = filter_lora_keys(lora_sd, blocktypes)

                if "lora_unet_single_transformer_blocks_0_attn_to_k.lora_up.weight" in lora_sd:
                    from .utils import convert_to_diffusers
                    lora_sd = convert_to_diffusers("lora_unet_", lora_sd)

                if not "transformer.single_transformer_blocks.0.attn_to.k.lora_A.weight" in lora_sd:
                    print(f"Converting LoRA weights from {l['path']} to diffusers format...")
                    lora_sd = _convert_hunyuan_video_lora_to_diffusers(lora_sd)

                lora_rank = None
                for key, val in lora_sd.items():
                    if "lora_B" in key or "lora_up" in key:
                        lora_rank = val.shape[1]
                        break
                if lora_rank is not None:
                    print(f"Merging rank {lora_rank} LoRA weights from {l['path']} with strength {l['strength']}")
                    adapter_name = l['path'].split("/")[-1].split(".")[0]
                    adapter_weight = l['strength']
                    transformer.load_lora_adapter(lora_sd, weight_name=l['path'].split("/")[-1], lora_rank=lora_rank, adapter_name=adapter_name)

                    adapter_list.append(adapter_name)
                    adapter_weights.append(adapter_weight)

                del lora_sd
                mm.soft_empty_cache()

            if adapter_list:
                transformer.set_adapters(adapter_list, weights=adapter_weights)
                if fuse:
                    if model_weight_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                        raise ValueError("Fusing LoRA doesn't work well with fp8 model weights. Please use a bf16 model file, or disable LoRA fusing.")
                    lora_scale = 1
                    transformer.fuse_lora(lora_scale=lora_scale)
                    transformer.delete_adapters(adapter_list)

            if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast" or quantization == "fp8_e5m2":
                params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
                for name, param in transformer.named_parameters():
                    # Make sure to not cast the LoRA weights to fp8.
                    if not any(keyword in name for keyword in params_to_keep) and not 'lora' in name:
                        param.data = param.data.to(after_lora_dtype)

        if quantization == "fp8_e4m3fn_fast":
            from .fp8_optimization import convert_fp8_linear
            convert_fp8_linear(transformer, base_dtype, params_to_keep=params_to_keep)

        DynamicSwapInstaller.install_model(transformer, device=device)

        if compile_args is not None:
            if compile_args["compile_single_blocks"]:
                for i, block in enumerate(transformer.single_transformer_blocks):
                    transformer.single_transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_double_blocks"]:
                for i, block in enumerate(transformer.transformer_blocks):
                    transformer.transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        pipe = {
            "transformer": transformer.eval(),
            "dtype": base_dtype,
        }
        return (pipe, )

def warped_clip_vision_load(ckpt_path):
    print("Reading: {}".format(ckpt_path))
    with open(ckpt_path, "rb") as file:
        clip_vision_temp = file.read()

    print("Loading Clip Vision Model...")
    sd = warped_load_torch_file(clip_vision_temp)
    print("Loading Clip Vision Model...Done")

    clip_vision_temp = None
    gc.collect()
    time.sleep(1)

    # sd = load_torch_file(ckpt_path)
    if "visual.transformer.resblocks.0.attn.in_proj_weight" in sd:
        return comfy.clip_vision.load_clipvision_from_sd(sd, prefix="visual.", convert_keys=True)
    else:
        return comfy.clip_vision.load_clipvision_from_sd(sd)

class WarpedCLIPVisionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("clip_vision"), ),
                             }}
    RETURN_TYPES = ("CLIP_VISION",)
    RETURN_NAMES = ("clip_vision",)
    FUNCTION = "load_clip"
    CATEGORY = "Warped/General/Loaders"

    def load_clip(self, clip_name):
        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_name)
        clip_vision = warped_clip_vision_load(clip_path)
        return (clip_vision,)

class WarpedVAELoader:
    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."):
                f1_taesd_dec = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        encoder_path = folder_paths.get_full_path_or_raise("vae_approx", encoder)

        print("Reading vae_approx encoder: {}".format(encoder_path))

        with open(encoder_path, "rb") as file:
            vae_temp = file.read()

        enc = warped_load_torch_file(vae_temp)
        vae_temp = None

        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        decoder_path = folder_paths.get_full_path_or_raise("vae_approx", decoder)

        print("Reading vae_approx encoder: {}".format(decoder_path))

        with open(encoder_path, "rb") as file:
            vae_temp = file.read()

        dec = warped_load_torch_file(vae_temp)
        vae_temp = None

        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (s.vae_list(), )}}
    RETURN_TYPES = ("VAE", "DICT", )
    RETURN_NAMES = ("vae", "vae_state_dict", )
    FUNCTION = "load_vae"

    CATEGORY = "Warped/General/Loaders"

    #TODO: scale factor?
    def load_vae(self, vae_name):
        if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            sd = self.load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)

            print("Reading VAE: {}".format(vae_path))
            with open(vae_path, "rb") as file:
                vae_temp = file.read()

            sd = warped_load_torch_file(vae_temp)
            vae_temp = None

        vae = comfy.sd.VAE(sd=sd)

        gc.collect()
        time.sleep(1)

        vae.throw_exception_if_invalid()
        return (vae, sd, )

class WarpedNumericalConversion:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                        "int_value": ("INT", {"default": None, "forceInput": True}),
                        "float_value": ("FLOAT", {"default": None, "forceInput": True}),
                        "number_value": ("NUMBER", {"default": None, "forceInput": True}),
                        "bool_value": ("BOOL", {"default": None, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT","FLOAT", "NUMBER", "STRING", )
    RETURN_NAMES = ("int","float", "number", "string", )
    OUTPUT_NODE = True
    FUNCTION = "int_to_number"
    CATEGORY = "Warped/General/Utils"

    def int_to_number(self, int_value=None, float_value=None, number_value=None, bool_value=None):
        value = None

        if (int_value == None) and (float_value == None) and (number_value == None) and (bool_value == None):
            raise ValueError("WarpedNumericalConversion: All inputs are None. Nothing to convert.")

        i = 0

        if not int_value is None:
            i += 1
            value = int_value

        if not float_value is None:
            i += 1
            value = float_value

        if not number_value is None:
            i += 1
            value = number_value

        if not bool_value is None:
            i += 1
            value = bool_value

        if i > 1:
            raise ValueError("WarpedNumericalConversion: More than one type of value inputs simultaneously is not supported.")

        if not value is None:
            return {"ui": {"string": ["{}".format(value),]}, "result": (int(value), float(value), value, "{}".format(value),), }

        return {"ui": {"string": ["{}".format(0),]}, "result": (int(0), float(0), value, "0",), }

class WarpedLoraReSave:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "save_path": ("STRING", {"default": get_default_gen_output_path()}),
                "trigger_words": ("STRING", {"default": ""}),
                "additional_info": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_lora"
    CATEGORY = "Warped/General/Lora"
    DESCRIPTION = "Re-saves LORA with new metadata."

    def get_metadata(self, lora_path, trigger_words, additional_info):
        model_config_path = os.path.join(script_directory, "your_metadata.json")
        model_config_path = os.path.abspath(model_config_path)

        print("metadata config file path: {}".format(model_config_path))

        try:
            with open(model_config_path, "r") as f:
                metadata = json.load(f)

            if len(trigger_words) > 0:
                metadata["trigger_words"] = trigger_words

            if len(additional_info) > 0:
                metadata["additional_info"] = additional_info

            return metadata
        except Exception as e:
            print("Exception attempting to read config file: {}".format(model_config_path))
            raise e

        return {"metadata": "No Metadata" }

    def load_lora(self, lora_name: str, save_path: str, trigger_words: str, additional_info: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not lora_name:
            return {"ui": {"tags": ["Nothing here to Re-save"]}}

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        lora_weights = comfy.utils.load_torch_file(lora_path)

        for key in lora_weights:
            lora_weights[key] = lora_weights[key].to(dtype=torch.bfloat16)

        metadata = self.get_metadata(lora_path, trigger_words, additional_info)

        temp_strings = save_path.split('\\')
        del temp_strings[len(temp_strings) - 1]

        print("Save Path: {}".format(save_path))

        save_folder = ""
        for temp_string in temp_strings:
            if len(save_folder) > 0:
                save_folder = "{}\\{}".format(save_folder, temp_string)
            else:
                save_folder = temp_string

        print("Save_folder: {}".format(save_folder))
        os.makedirs(save_folder, exist_ok = True)

        utils.save_torch_file(lora_weights, save_path, metadata=metadata)

        save_message = "Weights Saved To: {}".format(save_path)

        return {"ui": {"tags": [save_message]}}

class WarpedReverseImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Image batch to reverse."}),
            }
        }
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )
    FUNCTION = "reverse_images"
    CATEGORY = "Warped/General/Image"
    DESCRIPTION = "Reverses the order of the batch of images input into the node."

    def reverse_images(self, images=None):
        if images is None:
            raise ValueError("images input cannot be None.")

        images_tuple = torch.split(images, 1, dim=0)
        images_split = [item for item in images_tuple]

        images_split.reverse()

        resulting_images = None
        for temp_image in images_split:
            if not resulting_images is None:
                resulting_images = torch.cat((resulting_images, temp_image), 0)
            else:
                resulting_images = temp_image

        return (resulting_images, )

class WarpedCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", )
    RETURN_NAMES = ("model", "clip", "vae", )
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"
    CATEGORY = "Warped/General/Loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = self.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]

    def load_checkpoint_guess_config(self, ckpt_path, output_vae=True, output_clip=True, embedding_directory=None):
        print("Reading: {}".format(ckpt_path))
        with open(ckpt_path, "rb") as file:
            checkpoint_temp = file.read()

        print("Loading Checkpoint...")
        ckpt = warped_load_torch_file(checkpoint_temp)
        print("Loading Checkpoint...Done")
        checkpoint_temp = None

        gc.collect()
        time.sleep(1)

        out = comfy.sd.load_state_dict_guess_config(ckpt, output_vae, output_clip, embedding_directory)
        if out is None:
            raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_path))

        return out

class WarpedDatabase:
    def __init__(self, the_dictionary):
        self.data = the_dictionary

    def catExists(self, category):
        return category in self.data

    def keyExists(self, category, key):
        return category in self.data and key in self.data[category]

    def insert(self, category, key, value):
        if not isinstance(category, str) or not isinstance(key, str):
            cstr("Category and key must be strings").error.print()
            return

        if category not in self.data:
            self.data[category] = {}
        self.data[category][key] = value
        self._save()

    def update(self, category, key, value):
        if category in self.data and key in self.data[category]:
            self.data[category][key] = value
            self._save()

    def updateCat(self, category, dictionary):
        self.data[category].update(dictionary)
        self._save()

    def get(self, category, key):
        return self.data.get(category, {}).get(key, None)

    def getDB(self):
        return self.data

    def insertCat(self, category):
        if not isinstance(category, str):
            cstr("Category must be a string").error.print()
            return

        if category in self.data:
            cstr(f"The database category '{category}' already exists!").error.print()
            return
        self.data[category] = {}
        self._save()

    def getDict(self, category):
        if category not in self.data:
            cstr(f"The database category '{category}' does not exist!").error.print()
            return {}
        return self.data[category]

warped_DB = WarpedDatabase({"custom_tokens": {}})

class TextTokens:
    def __init__(self):
        self.WDB = warped_DB
        if not self.WDB.getDB().__contains__('custom_tokens'):
            self.WDB.insertCat('custom_tokens')
        self.custom_tokens = self.WDB.getDict('custom_tokens')

        self.tokens = {
            '[time]': str(time.time()).replace('.','_'),
            '[hostname]': socket.gethostname(),
            '[cuda_device]': str(comfy.model_management.get_torch_device()),
            '[cuda_name]': str(comfy.model_management.get_torch_device_name(device=comfy.model_management.get_torch_device())),
        }

        if '.' in self.tokens['[time]']:
            self.tokens['[time]'] = self.tokens['[time]'].split('.')[0]

        try:
            self.tokens['[user]'] = os.getlogin() if os.getlogin() else 'null'
        except Exception:
            self.tokens['[user]'] = 'null'

    def addToken(self, name, value):
        self.custom_tokens.update({name: value})
        self._update()

    def removeToken(self, name):
        self.custom_tokens.pop(name)
        self._update()

    def format_time(self, format_code):
        return time.strftime(format_code, time.localtime(time.time()))

    def parseTokens(self, text):
        tokens = self.tokens.copy()
        if self.custom_tokens:
            tokens.update(self.custom_tokens)

        # Update time
        tokens['[time]'] = str(time.time())
        if '.' in tokens['[time]']:
            tokens['[time]'] = tokens['[time]'].split('.')[0]

        for token, value in tokens.items():
            if token.startswith('[time('):
                continue
            pattern = re.compile(re.escape(token))
            text = pattern.sub(value, text)

        def replace_custom_time(match):
            format_code = match.group(1)
            return self.format_time(format_code)

        text = re.sub(r'\[time\((.*?)\)\]', replace_custom_time, text)

        return text

    def _update(self):
        self.WDB.updateCat('custom_tokens', self.custom_tokens)

class WarpedSamplerScriptsBase:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "required": {
            },
            "optional": {
            }
        }

        arg_dict["optional"]["dummy"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        for i in range(self.num_batchs):
            arg_dict["optional"][f"batch_{i}"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        return arg_dict

    RETURN_TYPES = ("WARPEDSCRIPTS",)
    RETURN_NAMES = ("scripts", )
    FUNCTION = "do_scripts"

    CATEGORY = "Warped/General/Scripts"

    def do_scripts(self, **kwargs):
        scripts = {}

        temp_text = kwargs.get(f"dummy")

        if len(temp_text) > 0:
            scripts["dummy"] = self.get_tokens(temp_text)

        for i in range(12):
            temp_text = kwargs.get(f"batch_{i}")

            if len(temp_text) > 0:
                scripts["{}".format(i)] = self.get_tokens(temp_text)

        return (scripts, )

    def get_tokens(self, text):
        new_text = []
        for line in io.StringIO(text):
            if not line.strip().startswith('#'):
                new_text.append(line.replace("\n", ''))
        new_text = "\n".join(new_text)

        tokens = TextTokens()
        new_text = tokens.parseTokens(new_text)

        return new_text

class WarpedSamplerScripts5(WarpedSamplerScriptsBase):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        arg_dict = {
            "required": {
            },
            "optional": {
            }
        }

        arg_dict["optional"]["dummy"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        for i in range(5):
            arg_dict["optional"][f"batch_{i}"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        return arg_dict

    def do_scripts(self, **kwargs):
        scripts = {}

        temp_text = kwargs.get(f"dummy")

        if len(temp_text) > 0:
            scripts["dummy"] = self.get_tokens(temp_text)

        for i in range(5):
            temp_text = kwargs.get(f"batch_{i}")

            if len(temp_text) > 0:
                scripts["{}".format(i)] = self.get_tokens(temp_text)

        return (scripts, )

class WarpedSamplerScripts8(WarpedSamplerScriptsBase):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        arg_dict = {
            "required": {
            },
            "optional": {
            }
        }

        arg_dict["optional"]["dummy"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        for i in range(8):
            arg_dict["optional"][f"batch_{i}"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        return arg_dict

    def do_scripts(self, **kwargs):
        scripts = {}

        temp_text = kwargs.get(f"dummy")

        if len(temp_text) > 0:
            scripts["dummy"] = self.get_tokens(temp_text)

        for i in range(8):
            temp_text = kwargs.get(f"batch_{i}")

            if len(temp_text) > 0:
                scripts["{}".format(i)] = self.get_tokens(temp_text)

        return (scripts, )

class WarpedSamplerScripts12(WarpedSamplerScriptsBase):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        arg_dict = {
            "required": {
            },
            "optional": {
            }
        }

        arg_dict["optional"]["dummy"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        for i in range(12):
            arg_dict["optional"][f"batch_{i}"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        return arg_dict

    def do_scripts(self, **kwargs):
        scripts = {}

        temp_text = kwargs.get(f"dummy")

        if len(temp_text) > 0:
            scripts["dummy"] = self.get_tokens(temp_text)

        for i in range(12):
            temp_text = kwargs.get(f"batch_{i}")

            if len(temp_text) > 0:
                scripts["{}".format(i)] = self.get_tokens(temp_text)

        return (scripts, )

class WarpedSamplerScripts16(WarpedSamplerScriptsBase):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        arg_dict = {
            "required": {
            },
            "optional": {
            }
        }

        arg_dict["optional"]["dummy"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        for i in range(16):
            arg_dict["optional"][f"batch_{i}"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        return arg_dict

    def do_scripts(self, **kwargs):
        scripts = {}

        temp_text = kwargs.get(f"dummy")

        if len(temp_text) > 0:
            scripts["dummy"] = self.get_tokens(temp_text)

        for i in range(16):
            temp_text = kwargs.get(f"batch_{i}")

            if len(temp_text) > 0:
                scripts["{}".format(i)] = self.get_tokens(temp_text)

        return (scripts, )

class WarpedSamplerScripts20(WarpedSamplerScriptsBase):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        arg_dict = {
            "required": {
            },
            "optional": {
            }
        }

        arg_dict["optional"]["dummy"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        for i in range(20):
            arg_dict["optional"][f"batch_{i}"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        return arg_dict

    def do_scripts(self, **kwargs):
        scripts = {}

        temp_text = kwargs.get(f"dummy")

        if len(temp_text) > 0:
            scripts["dummy"] = self.get_tokens(temp_text)

        for i in range(20):
            temp_text = kwargs.get(f"batch_{i}")

            if len(temp_text) > 0:
                scripts["{}".format(i)] = self.get_tokens(temp_text)

        return (scripts, )

class WarpedSamplerScripts30(WarpedSamplerScriptsBase):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        arg_dict = {
            "required": {
            },
            "optional": {
            }
        }

        arg_dict["optional"]["dummy"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        for i in range(30):
            arg_dict["optional"][f"batch_{i}"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        return arg_dict

    def do_scripts(self, **kwargs):
        scripts = {}

        temp_text = kwargs.get(f"dummy")

        if len(temp_text) > 0:
            scripts["dummy"] = self.get_tokens(temp_text)

        for i in range(30):
            temp_text = kwargs.get(f"batch_{i}")

            if len(temp_text) > 0:
                scripts["{}".format(i)] = self.get_tokens(temp_text)

        return (scripts, )

class WarpedSamplerScripts40(WarpedSamplerScriptsBase):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        arg_dict = {
            "required": {
            },
            "optional": {
            }
        }

        arg_dict["optional"]["dummy"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        for i in range(40):
            arg_dict["optional"][f"batch_{i}"] = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})

        return arg_dict

    def do_scripts(self, **kwargs):
        scripts = {}

        temp_text = kwargs.get(f"dummy")

        if len(temp_text) > 0:
            scripts["dummy"] = self.get_tokens(temp_text)

        for i in range(40):
            temp_text = kwargs.get(f"batch_{i}")

            if len(temp_text) > 0:
                scripts["{}".format(i)] = self.get_tokens(temp_text)

        return (scripts, )

class WarpedLoadImages:
    def __init__(self, index=0):
        self.index = index
        self.output_dir = folder_paths.get_output_directory()
        self.previous_path = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["incremental_image", "random"], {"default": "incremental_image"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                "label": ("STRING", {"default": 'Batch 001', "multiline": False}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "periodic_sleep": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "suffix": ("STRING", {"default": '', "multiline": False})
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("filename", "index", "prefixorg", "prefixseg", "prefixmsk")

    FUNCTION = "load_batch_images"

    CATEGORY = "Warped/General/Image"

    def load_batch_images(self, path, index=0, mode="incremental_image", label='Batch 001', periodic_sleep=False, suffix=""):
        if not os.path.exists(path):
            return (None, )

        if path != self.previous_path:
            self.index = index
            self.previous_path = path

        if periodic_sleep:
            if (self.index > 0) and (self.index % 10 == 0):
                time.sleep(2)

        fl = self.BatchImageLoader(path, label, '*', index)
        new_paths = fl.image_paths

        retry = True

        try:
            filename = fl.image_paths[self.index]
        except:
            if retry:
                retry = False
                self.index = 0
                filename = fl.image_paths[self.index]

        print("Filename: {}".format(filename))

        temp_strings = filename.split(".")
        temp_fileend = temp_strings[len(temp_strings) - 1]

        temp_filename = ""

        i = 0
        while i < (len(temp_strings) - 1):
            temp_filename = "{}{}".format(temp_filename, temp_strings[i])
            i += 1

        tempFilename = "{}/{}_{}.{}"
        tempFilenamesuffix = "{}/{}_{}_{}.{}"

        tempStrings1 = temp_filename.split("\\")
        temp_filename = tempStrings1[len(tempStrings1) - 1]

        if len(suffix) < 1:
            prefixorg = tempFilename.format(self.output_dir, temp_filename, "org", temp_fileend)
            prefixseg = tempFilename.format(self.output_dir, temp_filename, "seg", temp_fileend)
            prefixmsk = tempFilename.format(self.output_dir, temp_filename, "msk", temp_fileend)
        else:
            prefixorg = tempFilenamesuffix.format(self.output_dir, temp_filename, "org", suffix, temp_fileend)
            prefixseg = tempFilenamesuffix.format(self.output_dir, temp_filename, "seg", suffix, temp_fileend)
            prefixmsk = tempFilenamesuffix.format(self.output_dir, temp_filename, "msk", suffix, temp_fileend)

        self.index += 1

        if self.index >= len(fl.image_paths):
            self.index = 0

        return filename, self.index - 1, prefixorg, prefixseg, prefixmsk,

    class BatchImageLoader:
        def __init__(self, directory_path, label, pattern, index):
            self.image_paths = []
            self.load_images(directory_path, pattern)
            self.image_paths.sort()

            self.index = index
            self.label = label

        def load_images(self, directory_path, pattern):
            for file_name in glob.glob(os.path.join(directory_path, pattern), recursive=True):
                if file_name.lower().endswith(ALLOWED_EXT):
                    abs_file_path = os.path.abspath(file_name)
                    self.image_paths.append(abs_file_path)

        def get_image_by_id(self, image_id):
            if image_id < 0 or image_id >= len(self.image_paths):
                cstr(f"Invalid image index `{image_id}`").error.print()
                return

            return self.image_paths[image_id]

        def get_next_image(self):
            if self.index >= len(self.image_paths):
                self.index = 0

            image_path = self.image_paths[self.index]
            self.index += 1

            if self.index == len(self.image_paths):
                self.index = 0

            cstr(f'{cstr.color.YELLOW}{self.label}{cstr.color.END} Index: {self.index}').msg.print()

            return image_path

        def get_current_image(self):
            if self.index >= len(self.image_paths):
                self.index = 0
            image_path = self.image_paths[self.index]

            return image_path

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

class WarpedSaveImageCaption:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image_path": ("STRING", {"forceInput": True}),
                     "caption": ("STRING", {"forceInput": True}),
                    },
                }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    OUTPUT_NODE = True
    FUNCTION = "save_caption"

    CATEGORY = "Warped/General/Captioning"

    def save_caption(self, image_path, caption):
        tempStrings = image_path.split('.')
        caption_path = tempStrings[0]

        i = 1

        while i < (len(tempStrings) - 1):
            caption_path = "{}.{}".format(caption_path, tempStrings[i])
            i += 1

        caption_path = "{}.txt".format(caption_path)

        with open(caption_path, 'w') as fp:
            fp.write(caption)

        return {"ui": {"string": [caption,]}, "result": (caption,)}

class WarpedModifyCaptionFile:
    def __init__(self, index=0):
        self.index = index
        self.previous_path = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"mode": (["incremental_caption", "random"], {"default": "incremental_caption"}),
                     "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                     "caption_path": ("STRING", {"forceInput": False}),
                     "find_text": ("STRING", {"forceInput": False}),
                     "replace_text": ("STRING", {"forceInput": False}),
                    },
                }

    RETURN_TYPES = ("STRING", "STRING", "STRING", )
    RETURN_NAMES = ("filename", "original", "modified", )
    OUTPUT_NODE = True
    FUNCTION = "load_batch_captions"

    CATEGORY = "Warped/General/Captioning"

    def load_batch_captions(self, mode="incremental_caption", index=0, caption_path="", find_text="", replace_text=""):
        if (len(caption_path) == 0) or (len(find_text) == 0) or (not os.path.exists(caption_path)):
            return ("None", "None", "None")

        if caption_path != self.previous_path:
            self.index = index
            self.previous_path = caption_path

        fl = self.BatchCaptionLoader(caption_path, '*', index)

        retry = True

        try:
            filename = fl.caption_paths[self.index]
        except:
            if retry:
                retry = False
                self.index = 0
                filename = fl.caption_paths[self.index]

        print("Filename: {}".format(filename))

        original, modified = self.find_replace(filename, find_text, replace_text)

        self.index += 1

        if self.index >= len(fl.caption_paths):
            self.index = 0

        return {"ui": {"string": [filename, original, modified]}, "result": (filename, original, modified)}

    def find_replace(self, filename, find_text, replace_text):
        with open(filename, "r") as fp:
            original = fp.read()

        print("\nOriginal Text: {}".format(original))

        modified = original.replace(find_text.strip(), replace_text.strip())

        print("Modified Text: {}".format(modified))

        with open(filename, "w") as fp:
            fp.write(modified)

        return (original, modified)

    class BatchCaptionLoader:
        def __init__(self, directory_path, pattern, index):
            self.caption_paths = []
            self.load_captions(directory_path, pattern)
            self.caption_paths.sort()
            self.index = index

        def load_captions(self, directory_path, pattern):
            for file_name in glob.glob(os.path.join(directory_path, pattern), recursive=True):
                if file_name.lower().endswith(ALLOWED_CAPTION_EXT):
                    abs_file_path = os.path.abspath(file_name)
                    self.caption_paths.append(abs_file_path)

        def get_caption_by_id(self, caption_id):
            if caption_id < 0 or caption_id >= len(self.caption_paths):
                cstr(f"Invalid caption index `{caption_id}`").error.print()
                return

            return self.caption_paths[caption_id]

        def get_next_caption(self):
            if self.index >= len(self.caption_paths):
                self.index = 0

            caption_path = self.caption_paths[self.index]
            self.index += 1

            if self.index == len(self.caption_paths):
                self.index = 0

            cstr(f'{cstr.color.YELLOW}{self.label}{cstr.color.END} Index: {self.index}').msg.print()

            return caption_path

        def get_current_caption(self):
            if self.index >= len(self.caption_paths):
                self.index = 0
            caption_path = self.caption_paths[self.index]

            return caption_path

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

class WarpedAddToCaption:
    def __init__(self, index=0):
        self.index = index
        self.previous_path = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"mode": (["incremental_caption", "random"], {"default": "incremental_caption"}),
                     "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                     "caption_path": ("STRING", {"forceInput": False}),
                     "ignore_if_contains": ("STRING", {"forceInput": False}),
                     "add_text": ("STRING", {"forceInput": False}),
                     "add_mode": (["prefix", "postfix"], {"default": "prefix"}),
                    },
                }

    RETURN_TYPES = ("STRING", "STRING", "STRING", )
    RETURN_NAMES = ("filename", "original", "modified", )
    OUTPUT_NODE = True
    FUNCTION = "load_batch_captions"

    CATEGORY = "Warped/General/Captioning"

    def load_batch_captions(self, mode="incremental_caption", index=0, caption_path="", ignore_if_contains="", add_text="", add_mode="prefix"):
        if (len(caption_path) == 0) or (len(add_text) == 0) or (not os.path.exists(caption_path)):
            return ("None", "None", "None")

        if caption_path != self.previous_path:
            self.index = index
            self.previous_path = caption_path

        fl = self.BatchCaptionLoader(caption_path, '*', index)

        retry = True

        try:
            filename = fl.caption_paths[self.index]
        except:
            if retry:
                retry = False
                self.index = 0
                filename = fl.caption_paths[self.index]

        print("Filename: {}".format(filename))

        original, modified = self.process_prompt(filename, ignore_if_contains, add_text, add_mode)

        self.index += 1

        if self.index >= len(fl.caption_paths):
            self.index = 0

        return {"ui": {"string": [filename, original, modified]}, "result": (filename, original, modified)}

    def process_prompt(self, filename, ignore_if_contains, add_text, add_mode):
        with open(filename, "r") as fp:
            original = fp.read()

        print("\nOriginal Text: {}".format(original))

        if len(ignore_if_contains) > 0:
            if original.find(ignore_if_contains) != -1:
                return (original, original)

        if add_mode == "prefix":
            modified = "{}, {}".format(add_text, original)
        else:
            modified = "{}, {}".format(original, add_text)

        print("Modified Text: {}".format(modified))

        with open(filename, "w") as fp:
            fp.write(modified)

        return (original, modified)

    class BatchCaptionLoader:
        def __init__(self, directory_path, pattern, index):
            self.caption_paths = []
            self.load_captions(directory_path, pattern)
            self.caption_paths.sort()
            self.index = index

        def load_captions(self, directory_path, pattern):
            for file_name in glob.glob(os.path.join(directory_path, pattern), recursive=True):
                if file_name.lower().endswith(ALLOWED_CAPTION_EXT):
                    abs_file_path = os.path.abspath(file_name)
                    self.caption_paths.append(abs_file_path)

        def get_caption_by_id(self, caption_id):
            if caption_id < 0 or caption_id >= len(self.caption_paths):
                cstr(f"Invalid caption index `{caption_id}`").error.print()
                return

            return self.caption_paths[caption_id]

        def get_next_caption(self):
            if self.index >= len(self.caption_paths):
                self.index = 0

            caption_path = self.caption_paths[self.index]
            self.index += 1

            if self.index == len(self.caption_paths):
                self.index = 0

            cstr(f'{cstr.color.YELLOW}{self.label}{cstr.color.END} Index: {self.index}').msg.print()

            return caption_path

        def get_current_caption(self):
            if self.index >= len(self.caption_paths):
                self.index = 0
            caption_path = self.caption_paths[self.index]

            return caption_path

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

class WarpedPromptConcat:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"prompt1": ("STRING", {"default": ""}),
                     "prompt2": ("STRING", {"default": "", "forceInput": True}),
                     "delimiter": ([",", "."], {"default": ","}),
                    },
                }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("prompt", )
    FUNCTION = "do_concat"

    CATEGORY = "Warped/General/Utils"

    def do_concat(self, prompt1="", prompt2="", delimiter=","):
        temp_prompt1 = prompt1.strip().strip("\n").strip("\0").strip("\r")
        temp_prompt2 = prompt2.strip().strip("\n").strip("\0").strip("\r")

        output_prompt = "{}{} {}".format(temp_prompt1, delimiter, temp_prompt2)

        print("output_prompt: {}".format(output_prompt))

        return (output_prompt,)

class WarpedPromptConcatExt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"prompt1": ("STRING", {"default": ""}),
                     "prompt2": ("STRING", {"default": "", "forceInput": True}),
                     "delimiter": ([",", "."], {"default": ","}),
                     "filter_json_file": ("STRING", {"default": ""})
                    },
                }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("prompt", )
    FUNCTION = "do_concat"

    CATEGORY = "Warped/General/Utils"

    def do_concat(self, prompt1="", prompt2="", delimiter=",", filter_json_file=""):
        if len(filter_json_file) < 1:
            filter_json_file = None

        temp_prompt1 = prompt1.strip().strip("\n").strip("\0").strip("\r")
        temp_prompt2 = prompt2.strip().strip("\n").strip("\0").strip("\r")

        output_prompt = "{}{} {}".format(temp_prompt1, delimiter, temp_prompt2)

        filter_list = []
        if not filter_json_file is None:
            if not os.path.exists(filter_json_file):
                raise ValueError("{} Not Found!".format(filter_json_file))

            # try:
            with open(filter_json_file) as fp:
                filter_list = json.load(fp)
            # except:
            #     print("Error Encountered reading filter_json_file. Returning unfiltered prompt.")
            #     pass

            for entry in filter_list:
                output_prompt = output_prompt.replace(entry["find"], entry["replace"])

        print("output_prompt: {}".format(output_prompt))

        return (output_prompt,)

class WarpedImageLossCalc:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"predicted": ("IMAGE", ),
                     "target": ("IMAGE", ),
                    },
                }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("loss_calc", )
    FUNCTION = "do_calc"

    CATEGORY = "Warped/General/Utils"

    def do_calc(self, predicted, target):
        loss = torch.nn.functional.mse_loss(predicted.to(dtype=torch.bfloat16), target.to(dtype=torch.bfloat16), reduction="none")
        loss = loss.mean()
        loss_calc = "{:5.4f}".format(loss)

        return (loss_calc,)

class WarpedUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                              "weight_dtype": (["default", "f16", "bf16", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "Warped/General/Loaders"

    def load_unet(self, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16
        elif weight_dtype == "f16":
            model_options["dtype"] = torch.float16

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)

        print("Reading: {} . . .".format(unet_path))
        with open(unet_path, "rb") as file:
            content = file.read()
        print("Reading: {} . . . Done!".format(unet_path))

        chkpt, metadata = warped_load_torch_file(content, return_metadata=True)

        for key in chkpt:
            print("Diffusiion Model Key: {}  |  Shape: {}".format(key, chkpt[key].shape))

        model = comfy.sd.load_diffusion_model_state_dict(chkpt, model_options=model_options, metadata=metadata)

        if model is None:
            logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, chkpt)))

        # model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)

class WarpedUNETConverter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                              "weight_dtype": (["default", "fp16", "bf16", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
                             }}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert_model"

    CATEGORY = "Warped/General/Conversion"

    def load_unet(self, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16
        elif weight_dtype == "fp16":
            model_options["dtype"] = torch.float16

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)

        print("Reading: {} . . .".format(unet_path))
        with open(unet_path, "rb") as file:
            content = file.read()
        print("Reading: {} . . . Done!".format(unet_path))

        chkpt, metadata = warped_load_torch_file(content, return_metadata=True)
        content = None
        new_chkpt = {}

        for key in chkpt:
            print("Diffusiion Model Key: {}  |  Shape: {}".format(key, chkpt[key].shape))
            new_chkpt[key] = chkpt[key].to(dtype=model_options["dtype"])

        return new_chkpt, metadata, unet_path

    def convert_model(self, unet_name, weight_dtype):
        unet_data, metadata, unet_path = self.load_unet(unet_name, weight_dtype)

        metadata = {"original_metadata": "{}".format(metadata)}

        save_path = "{}_{}{}".format(os.path.splitext(unet_path)[0], weight_dtype, os.path.splitext(unet_path)[1])

        utils.save_torch_file(unet_data, save_path, metadata=metadata)

        unet_data = None
        mm.soft_empty_cache()
        gc.collect()
        time.sleep(1)

        save_message = "Weights Saved To: {}".format(save_path)

        return {"ui": {"tags": [save_message]}}

class WarpedHunyuanLoraDoubleBlocksSwap:
    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "required": {
                "lora_model_1": (['None'] + folder_paths.get_filename_list("loras"),),
                "lora_model_2": (['None'] + folder_paths.get_filename_list("loras"),),
                "mainstrength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),
                "save_path": ("STRING", {"default": get_default_output_path()}),
                "save_new_lora": ("BOOLEAN", {"default": False}),
                "return_state_only": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "model": ("MODEL", {"default": None}),
                "state_dictionary": ("WARPEDSTATEDICT1", {"default": None}),
                "metadata_dict": ("WARPEDMETADICT", {"default": None}),
                "lora_2_state_dictionary": ("WARPEDSTATEDICT2", {"default": None}),
                "metadata_flush": ("BOOLEAN", {"default": False}),
                "verbose_messaging": ("BOOLEAN", {"default": False}),
            }
        }

        for i in range(20):
            arg_dict["required"][f"double_blocks.{i}."] = ("BOOLEAN", {"default": False})

        return arg_dict

    RETURN_TYPES = ("MODEL", "STRING", "WARPEDSTATEDICT1", "WARPEDMETADICT",)
    RETURN_NAMES = ("model", "metadata", "state_dict", "metadata_dict",)
    FUNCTION = "load_lora"
    CATEGORY = "Warped/Hunyuan/Mixers"
    OUTPUT_NODE = False
    DESCRIPTION = "LoRA, single blocks double blocks"

    def load_lora(self, lora_model_1, lora_model_2, mainstrength, save_path, save_new_lora=False, return_state_only=False, model=None, state_dictionary=None, metadata_dict=None,
                lora_2_state_dictionary=None, metadata_flush=False, verbose_messaging=False, **kwargs):
        if ((lora_model_1 is None) and (state_dictionary)) or (lora_model_2 is None):
            ValueError("Both LORA models must be a valid selection, or LORA 1 has to be replaced by optional state dictionary input.")

        from comfy.utils import load_torch_file, save_torch_file
        from comfy.sd import load_lora_for_models
        from comfy.lora import load_lora

        new_metadata = None
        metadata = None
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.25)

        if state_dictionary is None:
            lora_1, metadata = warped_load_lora_weights(lora_model_1, return_metadata=True)
        else:
            lora_1 = state_dictionary

            if not metadata_dict is None:
                metadata = metadata_dict
            else:
                metadata = {}

        if metadata_flush or (metadata is None):
            metadata = {}

        new_metadata = metadata.copy()

        if not lora_2_state_dictionary is None:
            lora_2 = lora_2_state_dictionary
        elif not lora_model_2 == "None":
            lora_2 = warped_load_lora_weights(lora_model_2)

        diffusers_lora_1 = convert_lora(lora_1, convert_to="diffusion_model")
        filtered_lora_1 = filter_lora_keys(diffusers_lora_1, "all")

        diffusers_lora_2 = convert_lora(lora_2, convert_to="diffusion_model")
        filtered_lora_2 = filter_lora_keys(diffusers_lora_2, "all")

        max_dimension = 0

        for key in lora_1:
            if lora_1[key].shape[0] < lora_1[key].shape[1]:
                temp_dimension = lora_1[key].shape[0]
            else:
                temp_dimension = lora_1[key].shape[1]

            if temp_dimension > max_dimension:
                max_dimension = temp_dimension

            break

        for key in lora_2:
            if lora_2[key].shape[0] < lora_2[key].shape[1]:
                temp_dimension = lora_2[key].shape[0]
            else:
                temp_dimension = lora_2[key].shape[1]

            if temp_dimension > max_dimension:
                max_dimension = temp_dimension

            break

        filtered_lora_1 = convert_lora_dimensions(max_dimension, filtered_lora_1)
        filtered_lora_2 = convert_lora_dimensions(max_dimension, filtered_lora_2)

        block_settings = {
            **kwargs,
        }

        swap_block_settings = {k: v for k, v in block_settings.items() if v is True}
        swap_blocks = []

        for key in swap_block_settings:
            if swap_block_settings[key]:
                temp_strings = key.split('.')

                if len(temp_strings) > 1:
                    swap_blocks.append(int(temp_strings[1]))

        for entry in swap_blocks:
            temp_search_val = ".{}.".format(entry)

            for key in filtered_lora_1:
                if (temp_search_val in key) and (key in filtered_lora_2):
                    filtered_lora_1[key] = filtered_lora_2[key]

        if len(swap_blocks) > 0:
            if (not "modified_loras" in new_metadata) and (not "tune_counter" in new_metadata):
                new_metadata = {"modified_loras": "{} and {}".format(lora_model_1, lora_model_2)}

                swap_block_data = ""

                for entry in swap_blocks:
                    swap_block_data = "{} {}".format(swap_block_data, entry).strip()

                new_metadata["replaced_blocks"] = swap_block_data
            else:
                new_metadata["modified_loras"] = "{} | {} and {}".format(new_metadata["modified_loras"], lora_model_1, lora_model_2)

                swap_block_data = ""

                for entry in swap_blocks:
                    swap_block_data = "{} {}".format(swap_block_data, entry).strip()

                if "replaced_blocks" in new_metadata:
                    new_metadata["replaced_blocks"] = "{}  |  {}".format(new_metadata["replaced_blocks"], swap_block_data)
                else:
                    new_metadata["replaced_blocks"] = "{}".format(swap_block_data)

        if verbose_messaging:
            print("\nLora Metadata: {}".format(new_metadata))

        if save_new_lora and (len(swap_blocks) > 0):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            for key in filtered_lora_1:
                filtered_lora_1[key] = filtered_lora_1[key].to(dtype=torch.bfloat16)

            print("Saving Model To: {}...".format(save_path))
            save_torch_file(filtered_lora_1, save_path, metadata=new_metadata)
            print("Saving Model To: {}...Done.".format(save_path))

        if (not return_state_only) and (not model is None):
            new_model, _ = load_lora_for_models(model, None, filtered_lora_1, mainstrength, 0)
            if new_model is not None:
                return (new_model, new_metadata, None, None, )

        return (model, new_metadata, filtered_lora_1, new_metadata, )

    @classmethod
    def IS_CHANGED(s, lora_model_1, lora_model_2, mainstrength, save_path, save_new_lora, return_state_only=False, model=None, state_dictionary=None, metadata_dict=None, lora_2_state_dictionary=None, metadata_flush=False, verbose_messaging=False, **kwargs):
        return f"{lora_model_1}_{lora_model_2}_{mainstrength}_{state_dictionary}_{metadata_dict}"

def get_base_layer_type(layer_key):
    layer_key = layer_key.replace(".weight", "")

    if "single" in layer_key:
        layer_key = layer_key.replace("diffusion_model.single_blocks.", "")
    else:
        layer_key = layer_key.replace("diffusion_model.double_blocks.", "")

    if "img_" in layer_key:
        temp_strings = layer_key.split("img_")
        layer_key = "img_{}".format(temp_strings[len(temp_strings) - 1])
    else:
        temp_strings = layer_key.split("txt_")
        layer_key = "txt_{}".format(temp_strings[len(temp_strings) - 1])

    return layer_key

class WarpedHunyuanLoraDoubleBlocksLayersBlend:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_model_1": (['None'] + folder_paths.get_filename_list("loras"),),
                "lora_model_2": (['None'] + folder_paths.get_filename_list("loras"),),
                "mainstrength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),
                "save_path": ("STRING", {"default": get_default_output_path()}),
                "save_new_lora": ("BOOLEAN", {"default": False}),
                "return_state_only": ("BOOLEAN", {"default": False}),
                "img_attn_proj_lora_A": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "img_attn_proj_lora_B": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "img_attn_qkv_lora_A": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "img_attn_qkv_lora_B": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "img_mlp_fc1_lora_A": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "img_mlp_fc1_lora_B": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "img_mlp_fc2_lora_A": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "img_mlp_fc2_lora_B": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "img_mod_linear_lora_A": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "img_mod_linear_lora_B": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "txt_attn_proj_lora_A": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "txt_attn_proj_lora_B": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "txt_attn_qkv_lora_A": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "txt_attn_qkv_lora_B": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "txt_mlp_fc1_lora_A": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "txt_mlp_fc1_lora_B": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "txt_mlp_fc2_lora_A": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "txt_mlp_fc2_lora_B": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "txt_mod_linear_lora_A": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "txt_mod_linear_lora_B": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "for_all_img_layers": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
                "for_all_txt_layers": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
            },
            "optional": {
                "model": ("MODEL", {"default": None}),
                "state_dictionary": ("WARPEDSTATEDICT1", {"default": None}),
                "metadata_dict": ("WARPEDMETADICT", {"default": None}),
                "lora_2_state_dictionary": ("WARPEDSTATEDICT2", {"default": None}),
                "metadata_flush": (["tuner_only", "full", "none"], {"default": "tuner_only"}),
                "discard_single_blocks": ("BOOLEAN", {"default": True}),
                "verbose_messaging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "WARPEDSTATEDICT1", "WARPEDMETADICT",)
    RETURN_NAMES = ("model", "metadata", "state_dict", "metadata_dict",)
    FUNCTION = "load_lora"
    CATEGORY = "Warped/Hunyuan/Mixers"
    OUTPUT_NODE = False
    DESCRIPTION = "LoRA, single blocks double blocks"

    def load_lora(self, lora_model_1, lora_model_2, mainstrength, save_path, save_new_lora=False, return_state_only=False, model=None, state_dictionary=None, metadata_dict=None,
                lora_2_state_dictionary=None, metadata_flush="tuner_only", discard_single_blocks=True, verbose_messaging=False,
                img_attn_proj_lora_A=0.00, img_attn_proj_lora_B=0.00, img_attn_qkv_lora_A=0.00, img_attn_qkv_lora_B=0.00, img_mlp_fc1_lora_A=0.00,
                img_mlp_fc1_lora_B=0.00, img_mlp_fc2_lora_A=0.00, img_mlp_fc2_lora_B=0.00, img_mod_linear_lora_A=0.00, img_mod_linear_lora_B=0.00, txt_attn_proj_lora_A=0.00,
                txt_attn_proj_lora_B=0.00, txt_attn_qkv_lora_A=0.00, txt_attn_qkv_lora_B=0.00, txt_mlp_fc1_lora_A=0.00, txt_mlp_fc1_lora_B=0.00, txt_mlp_fc2_lora_A=0.00,
                txt_mlp_fc2_lora_B=0.00, txt_mod_linear_lora_A=0.00, txt_mod_linear_lora_B=0.00, for_all_img_layers=0.00, for_all_txt_layers=0.00):

        if lora_model_1 is None or lora_model_2 is None:
            ValueError("Bother LORA models must be valid selections.")

        from comfy.utils import load_torch_file, save_torch_file
        from comfy.sd import load_lora_for_models
        from comfy.lora import load_lora

        use_for_all_img = False

        if Decimal(for_all_img_layers).compare(Decimal(0.0)) != 0:
            use_for_all_img = True

        if use_for_all_img:
            print("Using For All Img.")
        else:
            print("Not Using For All Img.")

        use_for_all_txt = False

        if Decimal(for_all_txt_layers).compare(Decimal(0.0)) != 0:
            use_for_all_txt = True

        if use_for_all_txt:
            print("Using For All Txt.")
        else:
            print("Not Using For All Txt.")

        if state_dictionary is None:
            lora_1, metadata = warped_load_lora_weights(lora_model_1, return_metadata=True)
            print("++++ Metadata: ++++\n{}".format(metadata))
        else:
            lora_1 = state_dictionary

            if not metadata_dict is None:
                metadata = metadata_dict
            else:
                metadata = {}

        if not lora_2_state_dictionary is None:
            lora_2 = lora_2_state_dictionary
        elif not lora_model_2 == "None":
            lora_2 = warped_load_lora_weights(lora_model_2)

        diffusers_lora_1 = convert_lora(lora_1, convert_to="diffusion_model")
        filtered_lora_1 = filter_lora_keys(diffusers_lora_1, "all")

        diffusers_lora_2 = convert_lora(lora_2, convert_to="diffusion_model")
        filtered_lora_2 = filter_lora_keys(diffusers_lora_2, "all")

        max_dimension = 0

        for key in lora_1:
            if lora_1[key].shape[0] < lora_1[key].shape[1]:
                temp_dimension = lora_1[key].shape[0]
            else:
                temp_dimension = lora_1[key].shape[1]

            if temp_dimension > max_dimension:
                max_dimension = temp_dimension

            break

        for key in lora_2:
            if lora_2[key].shape[0] < lora_2[key].shape[1]:
                temp_dimension = lora_2[key].shape[0]
            else:
                temp_dimension = lora_2[key].shape[1]

            if temp_dimension > max_dimension:
                max_dimension = temp_dimension

            break

        filtered_lora_1 = convert_lora_dimensions(max_dimension, filtered_lora_1)
        filtered_lora_2 = convert_lora_dimensions(max_dimension, filtered_lora_2)

        tune_data = {}
        sub_key = ""

        dummy_lora = filtered_lora_1.copy()

        for key in dummy_lora:
            if "single_blocks" in key:
                if discard_single_blocks:
                    del filtered_lora_1[key]

                continue

            do_mod = False

            if not use_for_all_img and "img_" in key:
                if "img_attn_proj.lora_A" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - img_attn_proj_lora_A
                        perc_2 = img_attn_proj_lora_A

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["img_attn_proj.lora_A"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "img_attn_proj.lora_B" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - img_attn_proj_lora_B
                        perc_2 = img_attn_proj_lora_B

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["img_attn_proj.lora_B"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "img_attn_qkv.lora_A" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - img_attn_qkv_lora_A
                        perc_2 = img_attn_qkv_lora_A

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["img_attn_qkv.lora_A"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "img_attn_qkv.lora_B" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - img_attn_qkv_lora_B
                        perc_2 = img_attn_qkv_lora_B

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["img_attn_qkv.lora_B"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "img_mlp.fc1.lora_A" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - img_mlp_fc1_lora_A
                        perc_2 = img_mlp_fc1_lora_A

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["img_mlp.fc1.lora_A"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "img_mlp.fc1.lora_B" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - img_mlp_fc1_lora_B
                        perc_2 = img_mlp_fc1_lora_B

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["img_mlp.fc1.lora_B"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "img_mlp.fc2.lora_A" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - img_mlp_fc2_lora_A
                        perc_2 = img_mlp_fc2_lora_A

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["img_mlp.fc2.lora_A"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "img_mlp.fc2.lora_B" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - img_mlp_fc2_lora_B
                        perc_2 = img_mlp_fc2_lora_B

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["img_mlp.fc2.lora_B"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "img_mod.linear.lora_A" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - img_mod_linear_lora_A
                        perc_2 = img_mod_linear_lora_A

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["img_mod.linear.lora_A"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "img_mod.linear.lora_B" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - img_mod_linear_lora_B
                        perc_2 = img_mod_linear_lora_B

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["img_mod.linear.lora_B"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
            elif "img_" in key:
                sub_key = key.replace(".weight", "")
                sub_key = sub_key.replace("diffusion_model.double_blocks.", "")

                temp_strings = sub_key.split('img_')
                sub_key = "img_{}".format(temp_strings[len(temp_strings) - 1])

                perc_1 = 1.0 - for_all_img_layers
                perc_2 = for_all_img_layers

                tune_data[sub_key] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                do_mod = True

            if not use_for_all_txt and "txt_" in key:
                if "txt_attn_proj.lora_A" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - txt_attn_proj_lora_A
                        perc_2 = txt_attn_proj_lora_A

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["txt_attn_proj.lora_A"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "txt_attn_proj.lora_B" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - txt_attn_proj_lora_B
                        perc_2 = txt_attn_proj_lora_B

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["txt_attn_proj.lora_B"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "txt_attn_qkv.lora_A" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - txt_attn_qkv_lora_A
                        perc_2 = txt_attn_qkv_lora_A

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["txt_attn_qkv.lora_A"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "txt_attn_qkv.lora_B" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - txt_attn_qkv_lora_B
                        perc_2 = txt_attn_qkv_lora_B

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["txt_attn_qkv.lora_B"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "txt_mlp.fc1.lora_A" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - txt_mlp_fc1_lora_A
                        perc_2 = txt_mlp_fc1_lora_A

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["txt_mlp.fc1.lora_A"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "txt_mlp.fc1.lora_B" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - txt_mlp_fc1_lora_B
                        perc_2 = txt_mlp_fc1_lora_B

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["txt_mlp.fc1.lora_B"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "txt_mlp.fc2.lora_A" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - txt_mlp_fc2_lora_A
                        perc_2 = txt_mlp_fc2_lora_A

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["txt_mlp.fc2.lora_A"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "txt_mlp.fc2.lora_B" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - txt_mlp_fc2_lora_B
                        perc_2 = txt_mlp_fc2_lora_B

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["txt_mlp.fc2.lora_B"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "txt_mod.linear.lora_A" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - txt_mod_linear_lora_A
                        perc_2 = txt_mod_linear_lora_A

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["txt_mod.linear.lora_A"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
                elif "txt_mod.linear.lora_B" in key:
                    if key in filtered_lora_2:
                        perc_1 = 1.0 - txt_mod_linear_lora_B
                        perc_2 = txt_mod_linear_lora_B

                        if Decimal(perc_1).compare(Decimal(1.0)) != 0:
                            tune_data["txt_mod.linear.lora_B"] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                            do_mod = True
            elif "txt_" in key:
                sub_key = key.replace(".weight", "")
                sub_key = sub_key.replace("diffusion_model.double_blocks.", "")

                temp_strings = sub_key.split('txt_')
                sub_key = "txt_{}".format(temp_strings[len(temp_strings) - 1])

                perc_1 = 1.0 - for_all_txt_layers
                perc_2 = for_all_txt_layers

                tune_data[sub_key] = "perc_1: {} | perc_2: {}".format(perc_1, perc_2)
                do_mod = True

            if (key in filtered_lora_1) and (key in filtered_lora_2) and do_mod:
                mod_1 = torch.mul(filtered_lora_1[key], perc_1)
                mod_2 = torch.mul(filtered_lora_2[key], perc_2)

                new_lora = torch.add(mod_1, mod_2)
            else:
                continue

        if (metadata_flush == "full") or (metadata is None):
            metadata = {}

        print("**** Metadata: ****\n{}".format(metadata))

        if ((not "tune_counter_1" in metadata) and (not "modified_loras" in metadata)) or (metadata_flush == "full"):
            metadata = {"modified_loras": "{} and {}".format(lora_model_1, lora_model_2)}
        else:
            metadata["modified_loras"] = "{}  |  {} and {}".format(metadata["modified_loras"], lora_model_1, lora_model_2)

        if len(tune_data.keys()) > 0:
            if (not "tune_counter" in metadata) or (not metadata_flush == "none"):
                metadata["tune_counter"] = "{}".format(1)
                metadata[f'{tune_data}_1'] = "{}".format(tune_data)
            else:
                tune_counter = int(metadata["tune_counter"]) + 1
                metadata["tune_counter"] = "{}".format(tune_counter)
                metadata[f'{tune_data}_{tune_counter}'] = "{}".format(tune_data)

        if verbose_messaging:
            print("\nLora Metadata: {}".format(metadata))

        if save_new_lora: # and (len(swap_blocks) > 0):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            for key in filtered_lora_1:
                filtered_lora_1[key] = filtered_lora_1[key].to(dtype=torch.bfloat16)

            print("Saving Model To: {}...".format(save_path))
            save_torch_file(filtered_lora_1, save_path, metadata=metadata)
            print("Saving Model To: {}...Done.".format(save_path))

        if (not return_state_only) and (not model is None):
            new_model, _ = load_lora_for_models(model, None, filtered_lora_1, mainstrength, 0)
            if new_model is not None:
                return (new_model, metadata, None, None, )

        return (model, metadata, filtered_lora_1, metadata, )

    @classmethod
    def IS_CHANGED(s, lora_model_1, lora_model_2, mainstrength, save_path, save_new_lora=False, return_state_only=False, model=None, state_dictionary=None, metadata_dict=None,
                lora_2_state_dictionary=None, metadata_flush="tuner_only", discard_single_blocks=True, verbose_messaging=False,
                img_attn_proj_lora_A=0.00, img_attn_proj_lora_B=0.00, img_attn_qkv_lora_A=0.00, img_attn_qkv_lora_B=0.00, img_mlp_fc1_lora_A=0.00, img_mlp_fc1_lora_B=0.00, img_mlp_fc2_lora_A=0.00,
                img_mlp_fc2_lora_B=0.00, img_mod_linear_lora_A=0.00, img_mod_linear_lora_B=0.00, txt_attn_proj_lora_A=0.00, txt_attn_proj_lora_B=0.00, txt_attn_qkv_lora_A=0.00, txt_attn_qkv_lora_B=0.00, txt_mlp_fc1_lora_A=0.00, txt_mlp_fc1_lora_B=0.00,
                txt_mlp_fc2_lora_A=0.00, txt_mlp_fc2_lora_B=0.00, txt_mod_linear_lora_A=0.00, txt_mod_linear_lora_B=0.00, for_all_img_layers=0.00, for_all_txt_layers=0.00):
        return f"{lora_model_1}_{lora_model_2}_{mainstrength}"

class WarpedHunyuanLoraDoubleBlocksRemoveLinear:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_model": (['None'] + folder_paths.get_filename_list("loras"),),
                "mainstrength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),
                "save_path": ("STRING", {"default": get_default_output_path()}),
                "save_new_lora": ("BOOLEAN", {"default": False}),
                "return_state_only": ("BOOLEAN", {"default": False}),
                "max_dimension": ([32, 64, 128], {"default": 128}),
            },
            "optional": {
                "model": ("MODEL", {"default": None}),
                "state_dictionary": ("WARPEDSTATEDICT1", {"default": None}),
                "metadata_dict": ("WARPEDMETADICT", {"default": None}),
                "metadata_flush": (["deletes_only", "full", "none"], {"default": "deletes_only"}),
                "discard_single_blocks": ("BOOLEAN", {"default": True}),
                "verbose_messaging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "WARPEDSTATEDICT1", "WARPEDMETADICT",)
    RETURN_NAMES = ("model", "metadata", "state_dict", "metadata_dict",)
    FUNCTION = "load_lora"
    CATEGORY = "Warped/Hunyuan/Mixers"
    OUTPUT_NODE = False
    DESCRIPTION = "LoRA, single blocks double blocks"

    def load_lora(self, lora_model, mainstrength, save_path, save_new_lora=False, return_state_only=False, max_dimension=128, model=None, state_dictionary=None, metadata_dict=None, metadata_flush="deletes_only",
                discard_single_blocks=True, verbose_messaging=False):

        if lora_model is None and state_dictionary is None:
            ValueError("Either lora_model or state_dictionary input must be valid selections")

        from comfy.utils import load_torch_file, save_torch_file
        from comfy.sd import load_lora_for_models
        from comfy.lora import load_lora

        if state_dictionary is None:
            lora, metadata = warped_load_lora_weights(lora_model, return_metadata=True)
        else:
            lora = state_dictionary

            if not metadata_dict is None:
                metadata = metadata_dict
            else:
                metadata = {}

        diffusers_lora = convert_lora(lora, convert_to="diffusion_model")
        filtered_lora = filter_lora_keys(diffusers_lora, "all")
        filtered_lora = convert_lora_dimensions(max_dimension, filtered_lora)

        tune_data = {}
        sub_key = ""

        dummy_lora = filtered_lora.copy()
        deleted_keys = {}

        for key in dummy_lora:
            if "single_blocks" in key:
                if discard_single_blocks:
                    temp_key = get_base_layer_type(key)

                    deleted_keys[temp_key] = "Deleted All"
                    del filtered_lora[key]
                elif "linear" in key:
                    temp_key = get_base_layer_type(key)

                    deleted_keys[temp_key] = "Deleted All"
                    del filtered_lora[key]

                continue

            if not "linear" in key:
                continue

            temp_key = get_base_layer_type(key)

            deleted_keys[temp_key] = "Deleted All"

            del filtered_lora[key]

        if metadata_flush == "full":
            metadata = {}

        if ((not "deleted_keys_1" in metadata) and (not "modified_loras" in metadata)) or (metadata_flush == "full"):
            metadata = {"modified_loras": "{}".format(lora_model)}
        else:
            metadata["modified_loras"] = "{}  |  {}".format(metadata["modified_loras"], lora_model)

        if len(deleted_keys.keys()) > 0:
            if (not "deleted_counter" in metadata) or (not metadata_flush == "none"):
                metadata["deleted_counter"] = "{}".format(1)
                metadata[f'deleted_data_1'] = "{}".format(deleted_keys)
            else:
                deleted_counter = int(metadata["deleted_counter"]) + 1
                metadata["deleted_counter"] = "{}".format(deleted_counter)
                metadata[f'deleted_data_{deleted_counter}'] = "{}".format(deleted_keys)

        if verbose_messaging:
            print("\nLora Metadata: {}".format(metadata))

        if save_new_lora: # and (len(swap_blocks) > 0):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            for key in filtered_lora:
                filtered_lora[key] = filtered_lora[key].to(dtype=torch.bfloat16)

            print(metadata)

            print("Saving Model To: {}...".format(save_path))
            save_torch_file(filtered_lora, save_path, metadata=metadata)
            print("Saving Model To: {}...Done.".format(save_path))

        if (not return_state_only) and (not model is None):
            new_model, _ = load_lora_for_models(model, None, filtered_lora, mainstrength, 0)
            if new_model is not None:
                return (new_model, metadata, None, None, )

        return (model, metadata, filtered_lora, metadata, )

    @classmethod
    def IS_CHANGED(s, lora_model, mainstrength, save_path, save_new_lora=False, return_state_only=False, max_dimension=128, model=None, state_dictionary=None, metadata_dict=None, metadata_flush="tuner_only", discard_single_blocks=True, verbose_messaging=False):
        return f"{lora_model}_{mainstrength}"

class WarpedHunyuanLoraDoubleBlocksModifyMultipleSegments:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_model": (['None'] + folder_paths.get_filename_list("loras"),),
                "source_lora": (['None'] + folder_paths.get_filename_list("loras"),),
                "mainstrength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),
                "save_path": ("STRING", {"default": get_default_output_path()}),
                "save_new_lora": ("BOOLEAN", {"default": False}),
                "return_state_only": ("BOOLEAN", {"default": False}),
                "segment_numbers": ("STRING", {"default": "2,23"}),
                "test_mode": (["zero_all", "perc_all", "by_block_num", "random_noise", "use_source", "add_source", "subtract_source", "add_noise", "subtract_noise"], {"default": "perc_all"}),
                "max_dimension": ([32, 64, 128], {"default": 128}),
            },
            "optional": {
                "model": ("MODEL", {"default": None}),
                "state_dictionary": ("WARPEDSTATEDICT1", {"default": None}),
                "metadata_dict": ("WARPEDMETADICT", {"default": None}),
                "source_state_dictionary": ("WARPEDSTATEDICT2", {"default": None}),
                "block_number": ([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], {"default": 0}),
                "percentage": ("FLOAT", {"default": 1.000, "min": 0.000, "max": 5.000, "step": 0.001}),
                "layer_type": (["all", "img", "txt"], {"default": "all"}),
                "discard_single_blocks": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "verbose_messaging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "WARPEDSTATEDICT1", "WARPEDMETADICT",)
    RETURN_NAMES = ("model", "metadata", "state_dict", "metadata_dict",)
    FUNCTION = "load_lora"
    CATEGORY = "Warped/Hunyuan/Mixers/Experimental"
    OUTPUT_NODE = False
    DESCRIPTION = "LoRA, single blocks double blocks"

    def load_lora(self, lora_model, source_lora, mainstrength, save_path, save_new_lora=False, return_state_only=False, segment_numbers="2,23", test_mode="perc_all", max_dimension=128, model=None, state_dictionary=None, metadata_dict=None,
                source_state_dictionary=None, block_number=0, percentage=1.000, layer_type="all", discard_single_blocks=True, seed=0, verbose_messaging=False):

        if lora_model is None and state_dictionary is None:
            raise ValueError("Either lora_model or state_dictionary input must be valid selections")

        if len(segment_numbers) < 1:
            raise ValueError("segment_numbers cannot be empty.")

        temp_seg_split = segment_numbers.split(',')

        use_segment_numbers = []

        for entry in temp_seg_split:
            # if range of segments
            if "-" in entry.strip():
                if (test_mode == "add_noise") or (test_mode == "subtract_noise"):
                    raise ValueError("Segment Ranges Not Supported For test_mode add_noise  or subtract_noise")

                temp_seg_split_2 = entry.split("-")

                start = int(temp_seg_split_2[0].strip())
                end = int(temp_seg_split_2[1].strip())

                if start > end:
                    raise ValueError("Beginning of segment range cannot be greater than the end of the segment range.")

                for i in range(start, end + 1):
                    use_segment_numbers.append(i)

                continue

            use_segment_numbers.append(int(entry.strip()))

        if verbose_messaging:
            print("\n**** Using Segment Numbers:\n{}".format(use_segment_numbers))

        for segment_number in use_segment_numbers:
            if segment_number > (max_dimension - 1):
                raise ValueError("segment_number cannot be greater than max_dimension - 1.")

            if (segment_number < 1) and (test_mode == "random_noise"):
                raise ValueError("segment_number cannot be less than 1 when using test_mode == random_noise.")

        from comfy.utils import save_torch_file
        from comfy.sd import load_lora_for_models
        from comfy.lora import load_lora

        if state_dictionary is None:
            lora, metadata = warped_load_lora_weights(lora_model, return_metadata=True)
        else:
            lora = state_dictionary

            if not metadata_dict is None:
                metadata = metadata_dict
            else:
                metadata = {}

        if not "modified_loras" in metadata:
            metadata["modified_loras"] = "{} and {}".format(lora_model, source_lora)
        else:
            metadata["modified_loras"] = "{}  |  {} and {}".format(metadata["modified_loras"], lora_model, source_lora)

        diffusers_lora = convert_lora(lora, convert_to="diffusion_model")

        if discard_single_blocks:
            filtered_lora = filter_lora_keys(diffusers_lora, "double_blocks")
        else:
            filtered_lora = filter_lora_keys(diffusers_lora, "all")

        filtered_lora = convert_lora_dimensions(max_dimension, filtered_lora)

        source_filtered_lora = None

        if (test_mode == "use_source") or (test_mode == "add_source") or (test_mode == "subtract_source"):
            if not source_state_dictionary is None:
                temp_source_lora = source_state_dictionary
            elif not source_lora == "None":
                source_lora_path = folder_paths.get_full_path("loras", source_lora)

                if not os.path.exists(source_lora_path):
                    raise Exception(f"Lora {source_lora} not found at {source_lora_path}")

                temp_source_lora = warped_load_lora_weights(source_lora)

            source_diffusers_lora = convert_lora(temp_source_lora, convert_to="diffusion_model")
            source_filtered_lora = filter_lora_keys(source_diffusers_lora, "double_blocks")
            source_filtered_lora = convert_lora_dimensions(max_dimension, source_filtered_lora)

        block_filter = ""

        if test_mode == "by_block_num":
            block_filter = ".{}.".format(block_number)

        if not "slice_manipulation_counter" in metadata:
            metadata["slice_manipulation_counter"] = "1"
            metadata_key = "slice_manipulation_1"
        else:
            metadata["slice_manipulation_counter"] = "{}".format(int(metadata["slice_manipulation_counter"]) + 1)
            metadata_key = f'slice_manipulation_{"{}".format(metadata["slice_manipulation_counter"])}'

        metadata[metadata_key] = {"seed": "{}".format(seed)}
        metadata[metadata_key]["lora_model"] = lora_model
        metadata[metadata_key]["test_mode"] = test_mode
        metadata[metadata_key]["segment_numbers"] = "{}".format(segment_numbers)
        metadata[metadata_key]["max_dimension"] = "{}".format(max_dimension)
        metadata[metadata_key]["block_number"] = "{}".format(block_number)
        metadata[metadata_key]["layer_type"] = "{}".format(layer_type)
        metadata[metadata_key]["percentage"] = "{}".format(percentage)
        metadata[metadata_key]["discard_single_blocks"] = "{}".format(discard_single_blocks)

        if (test_mode == "use_source") or (test_mode == "add_source") or (test_mode == "subtract_source"):
            metadata[metadata_key]["source_model"] = source_lora
        else:
            metadata[metadata_key]["source_model"] = None

        metadata[metadata_key] = "{}".format(metadata[metadata_key])

        for key in filtered_lora:
            if "single_blocks" in key:
                continue

            if ("layer_type" == "img") and ("txt_" in key):
                continue

            if ("layer_type" == "txt") and ("img_" in key):
                continue

            if filtered_lora[key].shape[0] < filtered_lora[key].shape[1]:
                use_length = filtered_lora[key].shape[0]
                test_length = 1
            else:
                use_length = int(int(filtered_lora[key].shape[0]) // int(filtered_lora[key].shape[1]))
                test_length = int(filtered_lora[key].shape[0])

            temp_tensor = torch.zeros_like(filtered_lora[key])

            if (test_mode == "zero_all") or ((len(block_filter) > 0) and (block_filter in key)):
                for segment_number in use_segment_numbers:
                    if use_length == test_length:
                        if (not segment_number == 0) and (not segment_number == (max_dimension - 1)):
                            temp_tensor[:segment_number,:] = filtered_lora[key][:segment_number,:]
                            temp_tensor[segment_number + 1:,:] = filtered_lora[key][segment_number + 1:,:]
                        elif segment_number == 0:
                            temp_tensor[segment_number + 1:,:] = filtered_lora[key][segment_number + 1:,:]
                        else:
                            temp_tensor[:segment_number,:] = filtered_lora[key][:segment_number,:]
                    else:
                        if (not segment_number == 0) and (not segment_number == (max_dimension - 1)):
                            temp_tensor[:segment_number * test_length,:] = filtered_lora[key][:segment_number * test_length,:]
                            temp_tensor[segment_number + test_length:,:] = filtered_lora[key][segment_number + test_length:,:]
                        elif segment_number == 0:
                            temp_tensor[segment_number + test_length:,:] = filtered_lora[key][segment_number + test_length:,:]
                        else:
                            temp_tensor[:segment_number * test_length,:] = filtered_lora[key][:segment_number * test_length,:]

                    filtered_lora[key] = temp_tensor.to(torch.bfloat16)
            elif test_mode == "perc_all":
                temp_perc_tensor = torch.zeros_like(filtered_lora[key])

                for segment_number in use_segment_numbers:
                    if use_length == test_length:
                        if (not segment_number == 0) and (not segment_number == (max_dimension - 1)):
                            temp_tensor[:segment_number,:] = filtered_lora[key][:segment_number,:]
                            temp_tensor[segment_number + 1:,:] = filtered_lora[key][segment_number + 1:,:]

                            temp_perc_tensor[segment_number - 1:segment_number,:] = filtered_lora[key][segment_number - 1:segment_number,:] * percentage
                        elif segment_number == 0:
                            temp_tensor[segment_number + 1:,:] = filtered_lora[key][segment_number + 1:,:]
                            temp_perc_tensor[:1,:] = filtered_lora[key][:1,:] * percentage
                        else:
                            temp_tensor[:segment_number,:] = filtered_lora[key][:segment_number,:]
                            temp_perc_tensor[segment_number:segment_number + 1,:] = filtered_lora[key][segment_number:segment_number + 1,:] * percentage
                    else:
                        if (not segment_number == 0) and (not segment_number == (max_dimension - 1)):
                            temp_tensor[:segment_number * test_length,:] = filtered_lora[key][:segment_number * test_length,:]
                            temp_tensor[segment_number + test_length:,:] = filtered_lora[key][segment_number + test_length:,:]
                            temp_perc_tensor[segment_number - test_length:segment_number,:] = filtered_lora[key][segment_number - test_length:segment_number,:] * percentage
                        elif segment_number == 0:
                            temp_tensor[segment_number + test_length:,:] = filtered_lora[key][segment_number + test_length:,:]
                            temp_perc_tensor[:test_length,:] = filtered_lora[key][:test_length,:] * percentage
                        else:
                            temp_tensor[:segment_number * test_length,:] = filtered_lora[key][:segment_number * test_length,:]
                            temp_perc_tensor[(segment_number * test_length) - test_length:(segment_number * test_length) + test_length,:] = filtered_lora[key][(segment_number * test_length) - test_length:(segment_number * test_length) + test_length,:] * percentage

                    temp_tensor = torch.add(temp_tensor, temp_perc_tensor)
                    filtered_lora[key] = temp_tensor.to(torch.bfloat16)
            elif test_mode == "random_noise":
                rnd = torch.manual_seed(seed)

                random_noise_latent = warped_prepare_noise(torch.zeros_like(filtered_lora[key]), seed, generator=rnd)
                temp_perc_tensor = torch.zeros_like(filtered_lora[key])

                for segment_number in use_segment_numbers:
                    if use_length == test_length:
                        if (not segment_number == 0) and (not segment_number == (max_dimension - 1)):
                            temp_tensor[:segment_number,:] = filtered_lora[key][:segment_number,:]
                            temp_tensor[segment_number + 1:,:] = filtered_lora[key][segment_number + 1:,:]

                            temp_perc_tensor[segment_number - 1:segment_number,:] = random_noise_latent[segment_number - 1:segment_number,:] * percentage
                        elif segment_number == 0:
                            temp_tensor[segment_number + 1:,:] = filtered_lora[key][segment_number + 1:,:]

                            temp_perc_tensor[:1,:] = random_noise_latent[:1,:] * percentage
                        else:
                            temp_tensor[:segment_number,:] = filtered_lora[key][:segment_number,:]

                            temp_perc_tensor[segment_number:segment_number + 1,:] = random_noise_latent[segment_number:segment_number + 1,:] * percentage
                    else:
                        if (not segment_number == 0) and (not segment_number == (max_dimension - 1)):
                            temp_tensor[:segment_number * test_length,:] = filtered_lora[key][:segment_number * test_length,:]
                            temp_tensor[segment_number + test_length:,:] = filtered_lora[key][segment_number + test_length:,:]

                            temp_perc_tensor[segment_number - test_length:segment_number,:] = random_noise_latent[segment_number - test_length:segment_number,:] * percentage
                        elif segment_number == 0:
                            temp_tensor[segment_number + test_length:,:] = filtered_lora[key][segment_number + test_length:,:]

                            temp_perc_tensor[:test_length,:] = random_noise_latent[:test_length,:] * percentage
                        else:
                            temp_tensor[:segment_number * test_length,:] = filtered_lora[key][:segment_number * test_length,:]

                            temp_perc_tensor[(segment_number * test_length) - test_length:(segment_number * test_length) + test_length,:] = random_noise_latent[(segment_number * test_length) - test_length:(segment_number * test_length) + test_length,:] * percentage

                    temp_tensor = torch.add(temp_tensor, temp_perc_tensor)
                    filtered_lora[key] = temp_tensor.to(torch.bfloat16)
            elif (test_mode == "add_noise") or (test_mode == "subtract_noise"):
                rnd = torch.manual_seed(seed)

                for segment_number in use_segment_numbers:
                    filtered_lora[key] = filtered_lora[key].to(device=torch.device("cuda"))
                    random_noise_latent = warped_prepare_noise(torch.zeros_like(filtered_lora[key]), seed, generator=rnd)
                    temp_perc_tensor = torch.zeros_like(filtered_lora[key])

                    if use_length == test_length:
                        if (not segment_number == 0) and (not segment_number == (max_dimension - 1)):
                            temp_perc_tensor[segment_number - 1:segment_number,:] = random_noise_latent[segment_number - 1:segment_number,:] * percentage
                        elif segment_number == 0:
                            temp_perc_tensor[:1,:] = random_noise_latent[:1,:] * percentage
                        else:
                            temp_perc_tensor[segment_number:segment_number + 1,:] = random_noise_latent[segment_number:segment_number + 1,:] * percentage
                    else:
                        if (not segment_number == 0) and (not segment_number == (max_dimension - 1)):
                            temp_perc_tensor[segment_number - test_length:segment_number,:] = random_noise_latent[segment_number - test_length:segment_number,:] * percentage
                        elif segment_number == 0:
                            temp_perc_tensor[:test_length,:] = random_noise_latent[:test_length,:] * percentage
                        else:
                            temp_perc_tensor[(segment_number * test_length) - test_length:(segment_number * test_length) + test_length,:] = random_noise_latent[(segment_number * test_length) - test_length:(segment_number * test_length) + test_length,:] * percentage

                    if test_mode == "add_noise":
                        temp_tensor = torch.add(filtered_lora[key], temp_perc_tensor)
                    else:
                        temp_tensor = torch.sub(filtered_lora[key], temp_perc_tensor)

                    filtered_lora[key] = temp_tensor.to(torch.bfloat16)
            elif test_mode == "use_source":
                if not source_filtered_lora is None:
                    if not key in source_filtered_lora:
                        continue

                    for segment_number in use_segment_numbers:
                        temp_perc_tensor = torch.zeros_like(filtered_lora[key])

                        if use_length == test_length:
                            if (not segment_number == 0) and (not segment_number == (max_dimension - 1)):
                                temp_tensor[:segment_number,:] = filtered_lora[key][:segment_number,:]
                                temp_tensor[segment_number + 1:,:] = filtered_lora[key][segment_number + 1:,:]
                                temp_perc_tensor[segment_number - 1:segment_number,:] = source_filtered_lora[key][segment_number - 1:segment_number,:] * percentage
                            elif segment_number == 0:
                                temp_tensor[segment_number + 1:,:] = filtered_lora[key][segment_number + 1:,:]
                                temp_perc_tensor[:1,:] = source_filtered_lora[key][:1,:] * percentage
                            else:
                                temp_tensor[:segment_number,:] = filtered_lora[key][:segment_number,:]
                                temp_perc_tensor[segment_number:segment_number + 1,:] = source_filtered_lora[key][segment_number:segment_number + 1,:] * percentage
                        else:
                            if (not segment_number == 0) and (not segment_number == (max_dimension - 1)):
                                temp_tensor[:segment_number * test_length,:] = filtered_lora[key][:segment_number * test_length,:]
                                temp_tensor[segment_number + test_length:,:] = filtered_lora[key][segment_number + test_length:,:]
                                temp_perc_tensor[segment_number - test_length:segment_number,:] = source_filtered_lora[key][segment_number - test_length:segment_number,:] * percentage
                            elif segment_number == 0:
                                temp_tensor[segment_number + test_length:,:] = filtered_lora[key][segment_number + test_length:,:]
                                temp_perc_tensor[:test_length,:] = source_filtered_lora[key][:test_length,:] * percentage
                            else:
                                temp_tensor[:segment_number * test_length,:] = filtered_lora[key][:segment_number * test_length,:]
                                temp_perc_tensor[(segment_number * test_length) - test_length:(segment_number * test_length) + test_length,:] = source_filtered_lora[key][(segment_number * test_length) - test_length:(segment_number * test_length) + test_length,:] * percentage

                        temp_tensor = torch.add(temp_tensor, temp_perc_tensor)
                        filtered_lora[key] = temp_tensor.to(torch.bfloat16)
            elif (test_mode == "add_source") or (test_mode == "subtract_source"):
                if not source_filtered_lora is None:
                    if not key in source_filtered_lora:
                        continue

                    for segment_number in use_segment_numbers:
                        temp_perc_tensor = torch.zeros_like(filtered_lora[key])

                        if use_length == test_length:
                            if (not segment_number == 0) and (not segment_number == (max_dimension - 1)):
                                temp_perc_tensor[segment_number - 1:segment_number,:] = source_filtered_lora[key][segment_number - 1:segment_number,:] * percentage
                            elif segment_number == 0:
                                temp_perc_tensor[:1,:] = source_filtered_lora[key][:1,:] * percentage
                            else:
                                temp_perc_tensor[segment_number:segment_number + 1,:] = source_filtered_lora[key][segment_number:segment_number + 1,:] * percentage
                        else:
                            if (not segment_number == 0) and (not segment_number == (max_dimension - 1)):
                                temp_perc_tensor[segment_number - test_length:segment_number,:] = source_filtered_lora[key][segment_number - test_length:segment_number,:] * percentage
                            elif segment_number == 0:
                                temp_perc_tensor[:test_length,:] = source_filtered_lora[key][:test_length,:] * percentage
                            else:
                                temp_perc_tensor[(segment_number * test_length) - test_length:(segment_number * test_length) + test_length,:] = source_filtered_lora[key][(segment_number * test_length) - test_length:(segment_number * test_length) + test_length,:] * percentage

                        if test_mode == "add_source":
                            temp_tensor = torch.add(filtered_lora[key], temp_perc_tensor)
                        else:
                            temp_tensor = torch.sub(filtered_lora[key], temp_perc_tensor)

                        filtered_lora[key] = temp_tensor.to(torch.bfloat16)
                else:
                    print("**** No Source LORA Provided. Unable to make modification. ****")

        if verbose_messaging:
            print("Tester Metadata: {}".format(metadata))

        if save_new_lora: # and (len(swap_blocks) > 0):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            for key in filtered_lora:
                filtered_lora[key] = filtered_lora[key].to(dtype=torch.bfloat16)

            print("Saving Model To: {}...".format(save_path))
            save_torch_file(filtered_lora, save_path, metadata=metadata)
            print("Saving Model To: {}...Done.".format(save_path))

        if (not return_state_only) and (not model is None):
            new_model, _ = load_lora_for_models(model, None, filtered_lora, mainstrength, 0)
            if new_model is not None:
                return (new_model, metadata, None, None, )

        return (model, metadata, filtered_lora, metadata, )


    @classmethod
    def IS_CHANGED(s, lora_model, source_lora, mainstrength, save_path, save_new_lora=False, return_state_only=False, segment_numbers="2,23", test_mode="perc_all", max_dimension=128, model=None, state_dictionary=None, metadata_dict=None,
                source_state_dictionary=None, block_number=0, percentage=1.000, layer_type="all", discard_single_blocks=True, seed=0, verbose_messaging=False):
        return f"{lora_model}_{mainstrength}"

class WarpedLoadHunyuanLoraWeightsByPrefix:
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

    RETURN_TYPES = ("WARPEDSTATEDICT1", "WARPEDSTATEDICT2", "WARPEDMETADICT", )
    RETURN_NAMES = ("state_dict_1", "state_dict_2", "metadata_dict", )
    FUNCTION = "load_batch_loras"

    CATEGORY = "Warped/Hunyuan/Lora"

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
            print("WarpedLoadHunyuanLoraWeightsByPrefix: Filename: {}  |  Full File Path: {}".format(filename, full_filename))

            lora, metadata = warped_load_lora_weights(filename, return_metadata=True)

            return (lora, lora, metadata,)
        except:
            self.index = 0
            retry = True

        if retry:
            retry = False
            filename, full_filename = self.do_the_load(path, lora_prefix, index)
            print("WarpedLoadHunyuanLoraWeightsByPrefix: Retrying: Filename: {}  |  Full File Path: {}".format(filename, full_filename))

            lora, metadata = warped_load_lora_weights(filename, return_metadata=True)

            return (lora, lora, metadata,)

        return (None, None, None)

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

class WarpedAnyType(str):
    """A special type that always compares equal to any value."""

    def __ne__(self, __value: object) -> bool:
        return False

WARPEDANYTYPE = WarpedAnyType("*")

class WarpedContinueWorkflowManual:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (WARPEDANYTYPE,),
                "continue_workflow": ("BOOLEAN", {"default": True})
            },
            "hidden": {
                "id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (WARPEDANYTYPE, )
    RETURN_NAMES = ( "any", )
    FUNCTION = "execute"
    CATEGORY = "Warped/General/Utils"
    OUTPUT_NODE = True

    def execute(self, any=None, continue_workflow=True, id=None):
        if not continue_workflow:
            # print(f"Cancelled workflow for {id}")
            raise comfy.model_management.InterruptProcessingException()

        return {"result": (any,)}

class WarpedContinueWorkflowAuto:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (WARPEDANYTYPE,),
                "continue_workflow": ("BOOL", {"default": True, "forceInput": True})
            },
            "hidden": {
                "id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (WARPEDANYTYPE, )
    RETURN_NAMES = ( "any", )
    FUNCTION = "execute"
    CATEGORY = "Warped/General/Utils"
    OUTPUT_NODE = True

    def execute(self, any=None, continue_workflow=True, id=None):
        if not continue_workflow:
            # print(f"Cancelled workflow for {id}")
            raise comfy.model_management.InterruptProcessingException()

        return {"result": (any,)}
