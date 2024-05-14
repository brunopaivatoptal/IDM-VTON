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
import src.tryon_pipeline as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from PIL import Image
from torch import nn
import transformers
import numpy as np
import accelerate
import diffusers
import torch
import json
import os 

logger = logging.get_logger(__name__)


class IdmVtonModel(nn.Module):
    """
    Simplified IDM VTON model class, compatible with training.
    The 'forward' method in this class is based on the 'src.tryon_pipeline' code, and uses 
    some of the methods in Kinyugo's scripts.
    """
    def __init__(self, ckpt_dir,
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
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 image_sz=(256, 256)):
        super().__init__()
        self.pipe = pl.StableDiffusionXLInpaintPipeline.from_pretrained(
                ckpt_dir,
                unet=unet,
                vae=vae,
                feature_extractor= CLIPImageProcessor(),
                text_encoder = text_encoder,
                text_encoder_2 = text_encoder_2,
                tokenizer = tokenizer,
                tokenizer_2 = tokenizer_2,
                scheduler = scheduler,
                image_encoder=image_encoder,
                torch_dtype=torch.float16,
        )
        self.image_sz = image_sz
        self._execution_device=device
        self.vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)

    
    def forward(self,batch):
        with torch.cuda.amp.autocast():
            num_prompts = batch["person_image"].shape[0]
            prompt_embeddings = self.encode_cloth_and_model_prompts(num_prompts, 
                                                                    batch["caption"], 
                                                                    batch["caption_cloth"])
            person_latents = self._image_to_latent(batch["person_image"])
            identity_latents = self._image_to_latent(batch["identity_image"])
            identity_mask_latents = self._mask_to_latent(batch["identity_mask"])
            densepose_latents = self._image_to_latent(batch["densepose_image"])
            
            encoder_hidden_states = self.image_encoder(batch["clip_image_encoder_garment_image"])
            
            timesteps = torch.randint(
                low=0,
                high=self.scheduler.config.num_train_timesteps,
                size=(person_latents.shape[0],),
                device=person_latents.device,
                dtype=torch.long,
                )
            noise = torch.randn_like(person_latents)
            noisy_person_latents = self.scheduler.add_noise(
                person_latents, noise, timesteps
            )
            
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents,pose_img], dim=1)
            
        model_pred = self.unet(
            garment_unet_input=batch["garment_image"],
            person_unet_input=person_unet_input,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
            )[0]
            
            
        
    def _image_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        return (
            self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
        )

    def _mask_to_latent(self, mask: torch.Tensor) -> torch.Tensor:
        size = tuple([s // self.vae_scale_factor for s in mask.shape[-2:]])
        return F.interpolate(mask, size=size)
    
    def encode_cloth_and_model_prompts(self,
                                       num_prompts,
                                       caption_model,
                                       caption_cloth,
                                       negative_prompts="monochrome, lowres, bad anatomy, worst quality, low quality",
                                       ):
        if not isinstance(caption_model, List):
            caption_model = [caption_model] * num_prompts
        if not isinstance(caption_cloth, List):
            caption_cloth = [caption_cloth] * num_prompts
        if not isinstance(negative_prompts, List):
            negative_prompts = [negative_prompts] * num_prompts
            
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                caption_model,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompts,
            )
            (
                prompt_embeds_c,
                _,
                _,
                _,
            ) = self.encode_prompt(
                caption_cloth,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompts,
            )
        return dict(model_prompt_embedding=prompt_embeds, 
                    cloth_prompt_embedding=prompt_embeds_c,
                    negative_prompt_embedding=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds)
        
    
    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
        
    
    @staticmethod 
    def load_from_initial_ckpt(folder="modelCheckpoints", 
                               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                               trainable=True,
                               image_sz=(256, 256)):
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
        
        model = IdmVtonModel(folder,
                             unet=unet,
                             vae=vae,
                             unet_encoder=UNet_Encoder,
                             feature_extractor= CLIPImageProcessor(),
                             text_encoder = text_encoder_one,
                             text_encoder_2 = text_encoder_two,
                             tokenizer = tokenizer_one,
                             tokenizer_2 = tokenizer_two,
                             scheduler = noise_scheduler,
                             image_encoder=image_encoder,
                             image_sz=image_sz,
                             device=device)
        return model