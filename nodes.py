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

import node_helpers

from torch import Tensor
from einops import repeat
from unittest.mock import patch

from comfy.ldm.flux.layers import timestep_embedding, apply_mod
from comfy.ldm.lightricks.model import precompute_freqs_cis
from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
from comfy.ldm.common_dit import rms_norm
from comfy.ldm.wan.model import sinusoidal_embedding_1d


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_VIDEO_EXT = ('mp4', 'flv', 'mov', 'avi', 'mpg', 'webm', 'mkv')

def get_offload_device():
    return torch.device("cpu")

def tensor2pil(image):
    _tensor_check_image(image)
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def pil2tensorSwap(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

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
        metadata["block_metadata"] = block_metadata
        metadata["max_dimension"] = "{}".format(max_dimension)

        print("Max Dimension: {}".format(max_dimension))

        save_message = ""

        for mixture_key in merge_mixtures:
            new_lora = {}
            output_filename = os.path.join(save_folder, "{}_{:05}.safetensors".format(model_prefix, int(mixture_key)))
            metadata["merge_mixture"] = "{}".format(merge_mixtures[mixture_key])

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

        metadata["block_metadata"] = "{}".format(block_metadata)
        metadata["max_dimension"] = "{}".format(max_dimension)

        print("Max Dimension: {}".format(max_dimension))

        save_message = ""

        for mixture_key in merge_mixtures:
            new_lora = {}
            output_filename = os.path.join(save_folder, "{}_{:05}.safetensors".format(model_prefix, int(mixture_key)))
            metadata["merge_mixture"] = "{}".format(merge_mixtures[mixture_key])

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

class WarpedBundleVideoImages:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"video_path": ("STRING", {"default": ""}),
                    "starting_index": ("INT", {"default": 0}),
                    "num_frames": ("INT", {"default": 61, "min": 5, "max": 1000001, "step": 4}),
                    },
                }
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", )
    RETURN_NAMES = ("image_batch", "first_image", "last_image", "num_frames",)
    FUNCTION = "generate"

    CATEGORY = "Warped/Video"

    def generate(self, video_path, starting_index, num_frames):
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('WarpedBundleVideoImages: width = %d' % width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('WarpedBundleVideoImages: height = %d' % height)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('WarpedBundleVideoImages: length = %d' % length)

        batched_images = None
        # print_it = True
        last_image = None
        first_image = None

        if starting_index > length:
            starting_index = 0

        if ((starting_index + num_frames) > length) or (num_frames == 0):
            num_frames = length - starting_index

        try:
            skip = 0
            if starting_index > 0:
                while(cap.isOpened()) and (skip < starting_index):
                    _, frameorig = cap.read()
                    skip += 1

            frame_count = 0
            while (cap.isOpened()) and (frame_count < num_frames):
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
                # if print_it:
                #     print("RsmBundleVideoImages: Sample temp_image Shape Before: {}".format(temp_image.shape))

                if len(temp_image.shape) < 4:
                    temp_image = temp_image.unsqueeze(0)

                # if print_it:
                #     print_it = False
                #     print("RsmBundleVideoImages: Sample temp_image Shape After: {}".format(temp_image.shape))

                if not batched_images is None:
                    batched_images = torch.cat((batched_images, temp_image), 0)
                else:
                    batched_images = temp_image
        except:
            print("WarpedBundleVideoImages: Exception During Video File Read.")
        finally:
            cap.release()

        print("WarpedBundleVideoImages: Batched Images Shape: {}".format(batched_images.shape))

        return (batched_images, first_image, last_image, int(batched_images.shape[0]))

class WarpedRsmBundleAllVideoImages:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"video_path": ("STRING", {"default": ""}),
                    },
                }
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", )
    RETURN_NAMES = ("image_batch", "first_image", "last_image", "num_frames",)
    FUNCTION = "generate"

    CATEGORY = "Warped/Video"

    def generate(self, video_path):
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('WarpedRsmBundleAllVideoImages: width = %d' % width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('WarpedRsmBundleAllVideoImages: height = %d' % height)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('WarpedRsmBundleAllVideoImages: length = %d' % length)

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
                temp_image = pil2tensorSwap(pil_image)

                if len(temp_image.shape) < 4:
                    temp_image = temp_image.unsqueeze(0)

                if first_image is None:
                    first_image = temp_image

                last_image = temp_image

                if not batched_images is None:
                    batched_images = torch.cat((batched_images, temp_image), 0)
                else:
                    batched_images = temp_image

                num_frames += 1
        except:
            print("WarpedRsmBundleAllVideoImages: Exception During Video File Read.")
        finally:
            cap.release()

        return (batched_images, first_image, last_image, num_frames, )

def augmentation_add_noise(image, noise_aug_strength, seed):
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
                    },
                }
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", )
    RETURN_NAMES = ("image_batch", "first_image", "last_image", "num_frames",)
    FUNCTION = "generate"

    CATEGORY = "Warped/Video"

    def generate(self, video_path, starting_index, num_frames):
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('WarpedBundleVideoImages: width = %d' % width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('WarpedBundleVideoImages: height = %d' % height)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('WarpedBundleVideoImages: length = %d' % length)

        batched_images = None
        # print_it = True
        last_image = None
        first_image = None

        if starting_index > length:
            starting_index = 0

        if ((starting_index + num_frames) > length) or (num_frames == 0):
            num_frames = length - starting_index

        try:
            skip = 0
            if starting_index > 0:
                while(cap.isOpened()) and (skip < starting_index):
                    _, frameorig = cap.read()
                    skip += 1

            frame_count = 0
            while (cap.isOpened()) and (frame_count < num_frames):
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
                # if print_it:
                #     print("RsmBundleVideoImages: Sample temp_image Shape Before: {}".format(temp_image.shape))

                if len(temp_image.shape) < 4:
                    temp_image = temp_image.unsqueeze(0)

                # if print_it:
                #     print_it = False
                #     print("RsmBundleVideoImages: Sample temp_image Shape After: {}".format(temp_image.shape))

                if not batched_images is None:
                    batched_images = torch.cat((batched_images, temp_image), 0)
                else:
                    batched_images = temp_image
        except:
            print("WarpedBundleVideoImages: Exception During Video File Read.")
        finally:
            cap.release()

        print("WarpedBundleVideoImages: Batched Images Shape: {}".format(batched_images.shape))

        return (batched_images, first_image, last_image, int(batched_images.shape[0]), )

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
            if is_long_side:
                if width <= height:
                    image = self.scale_to_side(image, height, True)
                else:
                    image = self.scale_to_side(image, width, True)
            else:
                if width >= height:
                    image = self.scale_to_side(image, height, False)
                else:
                    image = self.scale_to_side(image, width, False)

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
                    },
                }

    CATEGORY = "Warped/Image"

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )

    FUNCTION = "scale_image"

    def scale_image(self, image, length=1024, scale_to="long_side"):
        img = tensor2pilSwap(image)
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

        newImage = pil2tensorSwap([tempImage])

        return newImage,

#
