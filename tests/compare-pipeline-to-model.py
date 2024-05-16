# -*- coding: utf-8 -*-
"""
Special requirement:
Requires diffusers==0.25.1
"""
from transformers import AutoTokenizer, PretrainedConfig,CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLControlNetInpaintPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import ProjectConfiguration, set_seed
from src.unet_hacked_tryon import UNet2DConditionModel
from accelerate.logging import get_logger
from torchvision import transforms
from accelerate import Accelerator
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from packaging import version
from PIL import Image
import transformers
import torchvision
import numpy as np
import accelerate
import diffusers
import torch
import json
import os


noise_scheduler = DDPMScheduler.from_pretrained("modelCheckpoints", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(
    "modelCheckpoints",
    subfolder="vae",
    torch_dtype=torch.float16,
)
unet = UNet2DConditionModel.from_pretrained(
    "modelCheckpoints",
    subfolder="unet",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "modelCheckpoints",
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    "modelCheckpoints",
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)
text_encoder_one = CLIPTextModel.from_pretrained(
    "modelCheckpoints",
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    "modelCheckpoints",
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
tokenizer_one = AutoTokenizer.from_pretrained(
    "modelCheckpoints",
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    "modelCheckpoints",
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)

accelerator_project_config = ProjectConfiguration(project_dir="result")
accelerator = Accelerator(
    mixed_precision="bf16",
    project_config=accelerator_project_config,
)

# Freeze vae and text_encoder and set unet to trainable
unet.requires_grad_(False)
vae.requires_grad_(False)
image_encoder.requires_grad_(False)
UNet_Encoder.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
unet.eval()
UNet_Encoder.eval()


pipe = TryonPipeline.from_pretrained(
        "modelCheckpoints",
        unet=unet,
        vae=vae,
        feature_extractor= CLIPImageProcessor(),
        text_encoder = text_encoder_one,
        text_encoder_2 = text_encoder_two,
        tokenizer = tokenizer_one,
        tokenizer_2 = tokenizer_two,
        scheduler = noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder
pipe = pipe.to(accelerator.device)
self=pipe

train_data_dir = [x for x in 
                  [
                    r"E:\backups\toptal\pixelcut\virtual-try-on\viton_combined_annotated\viton_single_sample",
                    "/mnt/vdb/datasets/viton_combined_annotated/viton_single_sample",
                    r"E:\backups\toptal\pixelcut\virtual-try-on\viton_combined_annotated\viton_combined_annotated",
                    "/mnt/vdb/datasets/viton_combined_annotated/viton_combined_annotated"
                  ] if os.path.exists(x)][0]


from modules.VtonDataset import pil_to_tensor
from modules.dataloading import *
#ds = SDCNVTONDataset(data_dir=r"E:\backups\toptal\pixelcut\virtual-try-on\viton_combined_annotated\viton_combined_annotated",
#                     pretrained_processor_path="openai/clip-vit-large-patch14")

IMAGE_SIZE=(256, 256)

dm = SDCNVTONDataModule(
    SDCNVTONDataModuleConfig(
        train_data_dir=train_data_dir,
        val_data_dir="./",      
        image_size=IMAGE_SIZE
    )
)
dm.setup()
train_dl = dm.train_dataloader()

for sample in train_dl:
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            prompt = sample["caption"]
            num_prompts = sample['garment_image'].shape[0]
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    
            if not isinstance(prompt, List):
                prompt = [prompt] * num_prompts
            if not isinstance(negative_prompt, List):
                negative_prompt = [negative_prompt] * num_prompts
    
            image_embeds = sample['clip_image_encoder_garment_image']
    
            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
            
            
                prompt = sample["caption"]
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    
                if not isinstance(prompt, List):
                    prompt = [prompt] * num_prompts
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * num_prompts
                
                with torch.inference_mode():
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )
                prompt=None
                prompt_2: Optional[Union[str, List[str]]] = None
                mask_image: PipelineImageInput = None
                masked_image_latents: torch.FloatTensor = None
                height: Optional[int] = None
                width: Optional[int] = None
                padding_mask_crop: Optional[int] = None
                strength: float = 0.9999
                num_inference_steps: int = 50
                timesteps: List[int] = None
                denoising_start: Optional[float] = None
                denoising_end: Optional[float] = None
                guidance_scale: float = 7.5
                negative_prompt: Optional[Union[str, List[str]]] = None
                negative_prompt_2: Optional[Union[str, List[str]]] = None
                num_images_per_prompt: Optional[int] = 1
                eta: float = 0.0
                generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
                latents: Optional[torch.FloatTensor] = None
                ip_adapter_image: Optional[PipelineImageInput] = None
                output_type: Optional[str] = "pil"
                cloth =None
                pose_img = None
                text_embeds_cloth=None
                return_dict: bool = True
                cross_attention_kwargs: Optional[Dict[str, Any]] = None
                guidance_rescale: float = 0.0
                original_size: Tuple[int, int] = None
                crops_coords_top_left: Tuple[int, int] = (0, 0)
                target_size: Tuple[int, int] = None
                negative_original_size: Optional[Tuple[int, int]] = None
                negative_crops_coords_top_left: Tuple[int, int] = (0, 0)
                negative_target_size: Optional[Tuple[int, int]] = None
                aesthetic_score: float = 6.0
                negative_aesthetic_score: float = 2.5
                clip_skip: Optional[int] = None
                pooled_prompt_embeds_c=None
                callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None
                callback_on_step_end_tensor_inputs: List[str] = ["latents"]
                kwargs={}
                                
                generator = torch.Generator(pipe.device).manual_seed(42)
                prompt_embeds=prompt_embeds.to(accelerator.device).half()
                negative_prompt_embeds=negative_prompt_embeds.to(accelerator.device).half()
                pooled_prompt_embeds=pooled_prompt_embeds.to(accelerator.device).half()
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(accelerator.device).half()
                num_inference_steps=40
                generator=generator
                strength = 1.0
                pose_img = sample['densepose_image'].to(accelerator.device).half()
                text_embeds_cloth=prompt_embeds_c.to(accelerator.device).half()
                cloth = sample["garment_image"].to(accelerator.device).half()
                mask_image=sample['cloth_mask'].to(accelerator.device).half()
                image=(sample['person_image'].to(accelerator.device).half()+1.0)/2.0
                height=IMAGE_SIZE[1]
                width=IMAGE_SIZE[0]
                guidance_scale=2.0
                ip_adapter_image = image_embeds
                
                break