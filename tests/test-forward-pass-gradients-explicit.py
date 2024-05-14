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

accelerator_project_config = ProjectConfiguration(project_dir="result")
accelerator = Accelerator(
    mixed_precision="bf16",
    project_config=accelerator_project_config,
)

model = IdmVtonModel.load_from_initial_ckpt("modelCheckpoints")
model=model.to(accelerator.device)
self=model
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
train_dl, model = accelerator.prepare(train_dl, model)
for batch in train_dl:
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            caption_model = sample["caption"]
            caption_cloth = sample["caption_cloth"]
            num_prompts = sample['garment_image'].shape[0]
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            image_embeds = sample['clip_image_encoder_garment_image']
            break 
            
            generator = torch.Generator(accelerator.device).manual_seed(42)
            images = model(
                caption_model=caption_model,
                negative_prompts=negative_prompts,
                num_inference_steps=40,
                strength = 1.0,
                densepose_image = sample['densepose_image'].to(accelerator.device),
                caption_cloth=caption_cloth,
                cloth = sample["garment_image"].to(accelerator.device),
                mask_image=sample['cloth_mask'].to(accelerator.device),
                person_image=(sample['person_image'].to(accelerator.device)+1.0)/2.0, 
                guidance_scale=2.0,
                ip_adapter_image = image_embeds,
            )[0]

    
            for i in range(len(images)):
                x_sample = pil_to_tensor(images[i])
                torchvision.utils.save_image(x_sample,os.path.join("results",sample['caption'][i] + ".png"))
                
            break
        
