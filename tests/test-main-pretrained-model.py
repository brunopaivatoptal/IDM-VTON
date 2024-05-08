# -*- coding: utf-8 -*-
"""
Special requirement:
Requires diffusers==0.25.1
"""
from transformers import AutoTokenizer, PretrainedConfig,CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLControlNetInpaintPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import ProjectConfiguration, set_seed
from src.unet_hacked_tryon import UNet2DConditionModel
from accelerate.logging import get_logger
from torchvision import transforms
from accelerate import Accelerator
import torch.utils.data as data
import torch.nn.functional as F
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
).to(accelerator.device)
pipe.unet_encoder = UNet_Encoder



with torch.cuda.amp.autocast():
    with torch.no_grad():
        for sample in test_dataloader:
            img_emb_list = []
            for i in range(sample['cloth'].shape[0]):
                img_emb_list.append(sample['cloth'][i])
            
            prompt = sample["caption"]

            num_prompts = sample['cloth'].shape[0]                                        
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

            if not isinstance(prompt, List):
                prompt = [prompt] * num_prompts
            if not isinstance(negative_prompt, List):
                negative_prompt = [negative_prompt] * num_prompts

            image_embeds = torch.cat(img_emb_list,dim=0)

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
            
            
                prompt = sample["caption_cloth"]
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
                


                generator = torch.Generator(pipe.device).manual_seed(args.seed) if args.seed is not None else None
                images = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    strength = 1.0,
                    pose_img = sample['pose_img'],
                    text_embeds_cloth=prompt_embeds_c,
                    cloth = sample["cloth_pure"].to(accelerator.device),
                    mask_image=sample['inpaint_mask'],
                    image=(sample['image']+1.0)/2.0, 
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    ip_adapter_image = image_embeds,
                )[0]


            for i in range(len(images)):
                x_sample = pil_to_tensor(images[i])
                torchvision.utils.save_image(x_sample,os.path.join(args.output_dir,sample['im_name'][i]))
            