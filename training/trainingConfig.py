# -*- coding: utf-8 -*-
from dataclasses import dataclass 
from pathlib import Path

@dataclass
class trainingConfig:
    pretrained_model_name_or_path : Path=Path("modelCheckpoints")
    resolution : int = 256
    learning_rate : float = 1e-4
    weight_decay : float = 1e-2
    num_train_epochs : float = 100
    train_batch_size : int = 8
    noise_offset : float = None
    dataloader_num_workers : int = 0
    save_steps : int = 20_000
    mixed_precision : str = "bf16"
    output_dir : str = "results"
    logging_dir : str = "logs"
    report_to : str = "all"
    use_ema : bool = True
    noise_offset : bool = True
    lr_scheduler : str = "constant"
    lr_warmup_steps : int = 500
    snr_gamma : float = 0.1
    gradient_accumulation_steps : int = 4
    proportion_empty_prompts : float = 0.1
    trainig_data_possible_dirs : tuple =(
      r"E:\backups\toptal\pixelcut\virtual-try-on\viton_combined_annotated\viton_combined_annotated",
      "/mnt/vdb/datasets/viton_combined_annotated/viton_combined_annotated"
    )
