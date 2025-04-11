from .nodes import *

NODE_CLASS_MAPPINGS = {
    "WarpedHunyuanMultiLoraMerge": WarpedHunyuanMultiLoraMerge,
    "WarpedHunyuanMultiLoraAvgMerge": WarpedHunyuanMultiLoraAvgMerge,
    "WarpedHunyuanLoraMerge": WarpedHunyuanLoraMerge,
    "WarpedHunyuanLoraMergeMixer": WarpedHunyuanLoraMergeMixer,
    "WarpedHunyuanLoraAvgMerge": WarpedHunyuanLoraAvgMerge,
    "WarpedWanLoraMerge": WarpedWanLoraMerge,
    "WarpedSamplerCustomAdv": WarpedSamplerCustomAdv,
    "WarpedSamplerCustomAdvLatent": WarpedSamplerCustomAdvLatent,
    "WarpedCreateSpecialImageBatch": WarpedCreateSpecialImageBatch,
    "WarpedCreateEmptyImageBatch": WarpedCreateEmptyImageBatch,
    "WarpedBundleVideoImages": WarpedBundleVideoImages,
    "WarpedRsmBundleAllVideoImages": WarpedRsmBundleAllVideoImages,
    "WarpedImageNoiseAugmentation": WarpedImageNoiseAugmentation,
    "WarpedLeapfusionHunyuanI2V": WarpedLeapfusionHunyuanI2V,
    "WarpedSaveAnimatedPng": WarpedSaveAnimatedPng,
    "WarpedUpscaleWithModel": WarpedUpscaleWithModel,
    "WarpedLoadVideosBatch": WarpedLoadVideosBatch,
    "WarpedWanImageToVideo": WarpedWanImageToVideo,
    "WarpedTeaCache": WarpedTeaCache,
    "WarpedWanLoadAndEditLoraBlocks": WarpedWanLoadAndEditLoraBlocks,
    "WarpedImageResize": WarpedImageResize,
    "WarpedImageScaleToSide": WarpedImageScaleToSide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WarpedHunyuanMultiLoraMerge": "Warped Hunyuan Multi Lora Merge",
    "WarpedHunyuanMultiLoraAvgMerge": "Warped Hunyuan Multi Lora Avg Merge",
    "WarpedHunyuanLoraMerge": "Warped Hunyuan Lora Merge",
    "WarpedHunyuanLoraMergeMixer": "Warped Hunyuan Lora Merge Mixer",
    "WarpedHunyuanLoraAvgMerge": "Warped Hunyuan Lora Avg Merge",
    "WarpedWanLoraMerge": "Warped Wan Lora Merge",
    "WarpedSamplerCustomAdv": "Warped Sampler Custom Advanced (Image)",
    "WarpedSamplerCustomAdvLatent": "Warped Sampler Custom Advanced (Latent)",
    "WarpedCreateSpecialImageBatch": "Warped Create Special Image Batch",
    "WarpedCreateEmptyImageBatch": "Warped Create Empty Image Batch",
    "WarpedBundleVideoImages": "Warped Bundle Video Images",
    "WarpedRsmBundleAllVideoImages": "Warped Rsm Bundle All Video Images",
    "WarpedImageNoiseAugmentation": "Warped Image Noise Augmentation",
    "WarpedLeapfusionHunyuanI2V": "Warped Leapfusion Hunyuan I2V",
    "WarpedSaveAnimatedPng": "Warped Save Animated Png",
    "WarpedUpscaleWithModel": "Warped Upscale With Model",
    "WarpedLoadVideosBatch": "Warped Load Videos Batch",
    "WarpedWanImageToVideo": "Warped Wan Image To Video",
    "WarpedTeaCache": "Warped TeaCache",
    "WarpedWanLoadAndEditLoraBlocks": "Warped Wan Load And Edit Lora Blocks",
    "WarpedImageResize": "Warped Image Resize",
    "WarpedImageScaleToSide": "Warped Image Scale To Side",
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
