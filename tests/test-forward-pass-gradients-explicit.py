# -*- coding: utf-8 -*-
"""
Special requirement:
Requires diffusers==0.25.1
"""
import os
if os.path.exists("/mnt/vdb/virtual-try-on/IDM-VTON"):
    os.chdir("/mnt/vdb/virtual-try-on/IDM-VTON")
from transformers import AutoTokenizer, PretrainedConfig,CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLControlNetInpaintPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import ProjectConfiguration, set_seed
from src.unet_hacked_tryon import UNet2DConditionModel
from modules.IdmVtonModel import IdmVtonModel
from accelerate.logging import get_logger
from accelerate import Accelerator
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from packaging import version
from PIL import Image
import transformers
import numpy as np
import accelerate
import diffusers
import torch
import json

def show(x : torch.Tensor):
    xi = (x - x.min()) / (x.max() - x.min())
    plt.imshow(xi.cpu().detach().numpy()[0].transpose(1,2,0))
    plt.axis("off")
    plt.tight_layout()


train_data_dir = [x for x in 
                  [
                    r"E:\backups\toptal\pixelcut\virtual-try-on\viton_combined_annotated\viton_combined_annotated",
                    "/mnt/vdb/datasets/viton_combined_annotated/viton_combined_annotated"
                  ] if os.path.exists(x)][0]


model = IdmVtonModel.load_from_initial_ckpt("modelCheckpoints")
sum([x.numel() for x in model.parameters()]) ## Total of about 7bn parameters

from modules.dataloading import *

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
    break
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
                
                generator = torch.Generator(pipe.device).manual_seed(42)
                images = pipe(
                    prompt_embeds=prompt_embeds.to(accelerator.device),
                    negative_prompt_embeds=negative_prompt_embeds.to(accelerator.device),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(accelerator.device),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(accelerator.device),
                    num_inference_steps=40,
                    generator=generator,
                    strength = 1.0,
                    pose_img = sample['densepose_image'].to(accelerator.device),
                    text_embeds_cloth=prompt_embeds_c.to(accelerator.device),
                    cloth = sample["garment_image"].to(accelerator.device),
                    mask_image=sample['cloth_mask'].to(accelerator.device),
                    image=(sample['person_image'].to(accelerator.device)+1.0)/2.0, 
                    height=IMAGE_SIZE[1],
                    width=IMAGE_SIZE[0],
                    guidance_scale=2.0,
                    ip_adapter_image = image_embeds,
                )[0]
    
    
            for i in range(len(images)):
                x_sample = pil_to_tensor(images[i])
                torchvision.utils.save_image(x_sample,os.path.join("results",sample['caption'][i] + ".png"))
                
            break
        