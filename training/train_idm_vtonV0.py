import torch.nn.functional as F
from pathlib import Path
from torch import nn
import itertools
import argparse
import random
import torch
import time
import json
import os

import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from diffusers.training_utils import EMAModel, compute_snr
from diffusers import AutoencoderKL, DDPMScheduler#, UNet2DConditionModel
from diffusers.optimization import get_scheduler

## IDM VTON Modules
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel

## Custom Module Loading
from modules.dataloading import SDCNVTONDataModule, SDCNVTONDataModuleConfig
from training.trainingConfig import trainingConfig
from modules.DataLogger import DataLogger
import training.trainingUtils as ut


def main():
    args = trainingConfig()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder")
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet_encoder")
    img_feature_extractor = CLIPImageProcessor()

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), 
                            model_cls=UNet2DConditionModel, 
                            model_config=unet.config,
                            use_ema_warmup=True,
                            )
        ema_unet.to(accelerator.device)

    # freeze parameters of models to save more memory
    unet.requires_grad_(True).to(accelerator.device)
    vae.requires_grad_(False).to(accelerator.device)
    text_encoder.requires_grad_(False).to(accelerator.device)
    text_encoder_2.requires_grad_(False).to(accelerator.device)
    image_encoder.requires_grad_(False).to(accelerator.device)
    unet_encoder.requires_grad_(False).to(accelerator.device)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device) # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    unet_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    encoder_hid_proj = unet.encoder_hid_proj
    
    optimizer = torch.optim.AdamW(unet.parameters(),
                                  lr=args.learning_rate, 
                                  weight_decay=args.weight_decay)
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps
    )
    
    # dataloader
    train_data_dir = [x for x in args.trainig_data_possible_dirs if os.path.exists(x)][0]
    dataset = SDCNVTONDataModule(
        SDCNVTONDataModuleConfig(
            train_data_dir=train_data_dir,
            val_data_dir="./",      
            image_size=(args.resolution, args.resolution)
        )
    )
    dataset.setup()
    train_dl = dataset.train_dataloader()
    logger = DataLogger(columns=["epoch", "step", "objective"])
    
    unet, optimizer, train_dl, lr_scheduler, encoder_hid_proj = accelerator.prepare(
        unet, optimizer, train_dl, lr_scheduler, encoder_hid_proj
    )
    
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dl):
            #break
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    num_prompts = batch_sz = batch["person_image"].shape[0]
                    masked_image = batch["person_image"] * (batch["cloth_mask"] < 0.5)
                    pi = (batch["person_image"] + 1)/2
                    person_latents = ut.image_to_latent(pi, vae)
                    
                    #cloth_mask_latents = ut.mask_to_latent(batch["cloth_mask"], vae)
                    ## Use this more permissive mask from Kinyugo
                    ## The default mask does not accomodate for all clothing replacements...
                    cloth_mask_latents = ut.mask_to_latent(batch["identity_mask"], vae)
                    
                    densepose_latents = ut.image_to_latent(batch["densepose_image"], vae)
                    cloth_latents = ut.image_to_latent(batch["garment_image"], vae)
                    masked_image_latents = ut.image_to_latent(masked_image, vae)
                    ip_adapter_img = batch["clip_image_encoder_garment_image"]
                    latents = person_latents.to(accelerator.device)
                    ip_adapter_img = batch["clip_image_encoder_garment_image"]
                    
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents).to(accelerator.device, dtype=weight_dtype)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                with torch.no_grad():
                    image_embeds = image_encoder(batch["clip_image_encoder_garment_image"
                                                       ].to(accelerator.device,
                                                            dtype=weight_dtype)).image_embeds
                image_embeds_ = []
                drop_image_embeds = (torch.rand(batch_sz) < args.proportion_empty_prompts).to(torch.long)
                for image_embed, drop_image_embed in zip(image_embeds, drop_image_embeds):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)
            
                with torch.no_grad():
                    garment_prompt_embds = ut.encode_prompt(batch["caption_cloth"],
                                                            [text_encoder, text_encoder_2],
                                                            [tokenizer, tokenizer_2],
                                                            args.proportion_empty_prompts)
                    
                    model_prompt_embds = ut.encode_prompt(batch["caption"],
                                                            [text_encoder, text_encoder_2],
                                                            [tokenizer, tokenizer_2],
                                                            args.proportion_empty_prompts)
                    
                    down, encoded_reference_features = unet_encoder(cloth_latents.to(weight_dtype),
                                                                timesteps,
                                                                garment_prompt_embds[0],
                                                                return_dict=False)
                        
                # add cond
                target_size = original_size = torch.tensor([[args.resolution, args.resolution]] * batch_sz).to(accelerator.device)
                crop_coords_top_left = torch.tensor([[0, 0]] * batch_sz).to(accelerator.device)
                add_time_ids = [
                    original_size,
                    crop_coords_top_left,
                    target_size
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                
                image_embeds = ut.prepare_ip_adapter_image_embeds(ip_adapter_img, 
                                                                  accelerator.device, 
                                                                  1 * batch_sz, 
                                                                  encoder_hid_proj, 
                                                                  image_encoder, 
                                                                  img_feature_extractor,
                                                                  do_classifier_free_guidance=False)
                image_embeds = encoder_hid_proj(image_embeds).to(noisy_latents.dtype)
                
                unet_added_cond_kwargs = {"text_embeds": model_prompt_embds[1],
                                          "time_ids": add_time_ids,
                                          "image_embeds":image_embeds}
                
                latent_model_input = torch.cat([noisy_latents, 
                                                cloth_mask_latents, 
                                                masked_image_latents,
                                                densepose_latents], dim=1).to(weight_dtype)
                
                noise_residual_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=model_prompt_embds[0],
                    timestep_cond=None,
                    added_cond_kwargs=unet_added_cond_kwargs,
                    garment_features=encoded_reference_features,
                    return_dict=False,
                )[0]
                
                loss = F.mse_loss(noise_residual_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    if args.use_ema:
                        ema_unet.step(unet.parameters())

                if accelerator.is_main_process:
                    if args.use_ema:
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    logger.add([epoch, global_step, avg_loss])
                    logger.log()
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                if accelerator.is_main_process:
                    trained_model = accelerator.unwrap_model(unet)
                    to_save = trained_model.state_dict()
                    torch.save(to_save, os.path.join(save_path, "person_unet_model.ckpt"))
                    print("Saved checkpoint !")
            
            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
