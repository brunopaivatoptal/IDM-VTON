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



