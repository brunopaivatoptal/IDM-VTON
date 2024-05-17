# -*- coding: utf-8 -*-
"""
Utilitary functions to make the training code more readable.
"""
from diffusers.models import ImageProjection
import torch.nn.functional as F
from torch import nn
import numpy as np
import random
import torch


def image_to_latent(image: torch.Tensor, vae) -> torch.Tensor:
    return (
        vae.encode(image.to(torch.float32)).latent_dist.sample() * vae.config.scaling_factor
    )

def mask_to_latent(mask: torch.Tensor, vae) -> torch.Tensor:
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    size = tuple([int(s // vae_scale_factor) for s in mask.shape[-2:]])
    return F.interpolate(mask, size=size)


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def encode_image(image, device, num_images_per_prompt, image_encoder, 
                 feature_extractor, output_hidden_states=None):
    dtype = next(image_encoder.parameters()).dtype
    # print(image.shape)
    if not isinstance(image, torch.Tensor):
        image = feature_extractor(image, return_tensors="pt").pixel_values

    image = image.to(device=device, dtype=dtype)
    if output_hidden_states:
        image_enc_hidden_states = image_encoder(image, output_hidden_states=True).hidden_states[-2]
        image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_enc_hidden_states = image_encoder(
            torch.zeros_like(image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
            num_images_per_prompt, dim=0
        )
        return image_enc_hidden_states, uncond_image_enc_hidden_states
    else:
        image_embeds = image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

def prepare_ip_adapter_image_embeds(ip_adapter_image, device, num_images_per_prompt, 
                                    encoder_hid_proj, image_encoder, feature_extractor,
                                    do_classifier_free_guidance=True):
    output_hidden_state = not isinstance(encoder_hid_proj, ImageProjection)
    image_embeds, negative_image_embeds = encode_image(
        ip_adapter_image, device, 1, image_encoder, feature_extractor, output_hidden_state
    )
    if do_classifier_free_guidance:
        image_embeds = torch.cat([negative_image_embeds, image_embeds])
        image_embeds = image_embeds.to(device)
    return image_embeds