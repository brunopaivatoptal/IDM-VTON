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
from diffusers.utils.torch_utils import randn_tensor
from modules.IdmVtonModel import IdmVtonModel
from accelerate.logging import get_logger
from accelerate import Accelerator
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from packaging import version
from tqdm import tqdm
from PIL import Image
import transformers
import numpy as np
import accelerate
import diffusers
import hashlib
import torch
import json

def show(x : torch.Tensor):
    xi = (x - x.min()) / (x.max() - x.min())
    plt.imshow(xi.cpu().detach().numpy()[0].transpose(1,2,0))
    plt.axis("off")
    plt.tight_layout()

def tensor_fingerprint(t : torch.Tensor) -> str:
    s = " ".join([str(x) for x in t.detach().ravel().round(decimals=3).cpu().numpy()])
    h = hashlib.sha1(s.encode("utf8")).hexdigest()
    return h

train_data_dir = [x for x in 
                  [
                    r"E:\backups\toptal\pixelcut\virtual-try-on\viton_combined_annotated\viton_single_sample",
                    "/mnt/vdb/datasets/viton_combined_annotated/viton_single_sample",
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
            caption_model = prompt = batch["caption"]
            caption_cloth = batch["caption_cloth"]
            num_prompts = batch['garment_image'].shape[0]
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            image_embeds = batch['clip_image_encoder_garment_image']
            generator = torch.Generator(accelerator.device).manual_seed(42)
            break
            
            timesteps = torch.Tensor([976, 951, 926, 901, 876, 851, 826, 801, 776, 751, 726, 701, 676, 651,
                    626, 601, 576, 551, 526, 501, 476, 451, 426, 401, 376, 351, 326, 301,
                    276, 251, 226, 201, 176, 151, 126, 101,  76,  51,  26,   1]).to(torch.long)
            noisy_person_latents = randn_tensor((4,4,32,32),
                                                device=batch['garment_image'].device,
                                                dtype=torch.float16)
            start = noisy_person_latents
            
            for ts in tqdm(timesteps):
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        noise_residual_pred, _, _ = model(batch, ts,
                                                          noisy_person_latents=noisy_person_latents,
                                                          return_latents=True)
                        noisy_person_latents = self.scheduler.step(noise_residual_pred, 
                                                                      ts.repeat(4).cpu()[0], 
                                                                      noisy_person_latents,
                                                                      return_dict=False)[0]
                
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    noisy_person_image = self.vae.decode(start.cuda()).sample
                    a = noisy_person_image
                    a = (a - a.min()) / (a.max() - a.min())
                    noisy_person_latents = self.vae.decode(noisy_person_latents.cuda()).sample
                    b = noisy_person_image
                    b = (b - b.min()) / (b.max() - b.min())
            plt.figure(figsize=(12,7))
            plt.subplot(1,3,1)
            plt.title("Noisy")
            plt.imshow(a[0].cpu().float().numpy().transpose(1,2,0))
            plt.axis("off")
            plt.subplot(1,3,2)
            plt.title("Denoised")
            plt.imshow(b[0].cpu().float().numpy().transpose(1,2,0))
            plt.axis("off")
            plt.subplot(1,3,3)
            plt.title("Person Img")
            plt.imshow(batch["person_image"][0].cpu().float().numpy().transpose(1,2,0))
            plt.axis("off")

    
            for i in range(len(images)):
                x_sample = pil_to_tensor(images[i])
                torchvision.utils.save_image(x_sample,os.path.join("results",sample['caption'][i] + ".png"))
                
            break
        
