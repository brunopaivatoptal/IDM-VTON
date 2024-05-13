# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, PretrainedConfig,CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLControlNetInpaintPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from diffusers.utils.import_utils import is_xformers_available
from diffusers.schedulers import KarrasDiffusionSchedulers
from src.unet_hacked_tryon import UNet2DConditionModel
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
import transformers
import numpy as np
import accelerate
import diffusers
import torch
import json
import os 



class IdmVtonModel(nn.Module):
    """
    Simplified IDM VTON model class, compatible with training.
    The 'forward' method in this class is based on the 'src.tryon_pipeline' code, and uses 
    some of the methods in Kinyugo's scripts.
    """
    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 text_encoder_2: CLIPTextModelWithProjection,
                 tokenizer: CLIPTokenizer,
                 tokenizer_2: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 unet_encoder: UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 image_encoder: CLIPVisionModelWithProjection = None,
                 feature_extractor: CLIPImageProcessor = None,
                 image_sz=(256, 256)):
        super().__init__()
        self.vae=vae
        self.text_encoder=text_encoder
        self.text_encoder_2=text_encoder_2
        self.tokenizer=tokenizer
        self.unet=unet
        self.scheduler=scheduler
        self.image_encoder=image_encoder
        self.feature_extractor=feature_extractor
        self.unet_encoder=unet_encoder
        self.image_sz = image_sz
    
    def forward(self,
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                densepose_image,
                text_embeds_cloth,
                cloth,
                mask_image,
                person_image,
                ip_adapter_image,
                strength = 1.0,
                num_inference_steps=40,
                guidance_scale=2.0,):
        pass
    
    def encode_prompt(self, prompt):
        pass
    
    @staticmethod 
    def load_from_initial_ckpt(folder="modelCheckpoints", trainable=True, image_sz=(256, 256)):
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
        
        if not trainable:
            # Freeze vae and text_encoder and set unet to trainable
            unet.requires_grad_(False) ## 2,991,238,468 parameters (2.9Bn)
            vae.requires_grad_(False) ## 83,653,863 parameters (83M)
            image_encoder.requires_grad_(False) ## 632,076,800 (632M)
            UNet_Encoder.requires_grad_(False) ## 2,562,218,244 (2.5Bn)
            text_encoder_one.requires_grad_(False) ## 123,060,480 (123M)
            text_encoder_two.requires_grad_(False) ## 694,659,840 (693M)
            unet.eval()
            UNet_Encoder.eval()
        
        model = IdmVtonModel(unet=unet,
                             vae=vae,
                             unet_encoder=UNet_Encoder,
                             feature_extractor= CLIPImageProcessor(),
                             text_encoder = text_encoder_one,
                             text_encoder_2 = text_encoder_two,
                             tokenizer = tokenizer_one,
                             tokenizer_2 = tokenizer_two,
                             scheduler = noise_scheduler,
                             image_encoder=image_encoder,
                             image_sz=image_sz)
        return model
        