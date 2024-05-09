# -*- coding: utf-8 -*-
"""
Captioning model:
    https://huggingface.co/docs/transformers/main/en/model_doc/idefics2
"""
from transformers import BlipProcessor, BlipForConditionalGeneration
from accelerate import Accelerator
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd 
import numpy as np 
import requests
import torch
import os 

root = [x for x in 
        [Path(r"E:\backups\toptal\pixelcut\virtual-try-on\viton-partial"),
         Path("/mnt/vdb/datasets/viton_combined_annotated/viton_combined_annotated")]
        if os.path.exists(x)][0]
BATCH_SZ=400

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

accelerator = Accelerator()

allFiles = [root/x for x in os.listdir(root)]
print("Captioning", len(allFiles))

batch = []

for i, f in enumerate(tqdm(allFiles, desc="Captioning...")):
    if os.path.exists(f/"caption/caption.txt"):
        continue
    if os.path.isdir(f):
        base = f/"garment"
        fn = os.listdir(base)
        image = Image.open(base/fn[0])
        batch.append((image, f))
        if len(batch) >= BATCH_SZ or (i == len(allFiles)-1):
            inputs = processor([b[0] for b in batch], return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs)
            output_text = [processor.decode(o, skip_special_tokens=True) 
                           for o in out]
            for fi, txt in zip(batch, output_text):
                folder = fi[1]
                os.makedirs(folder/"caption", exist_ok=True)
                with open(folder/"caption"/"caption.txt", "w") as file:
                    file.write(txt)
                
            plt.figure(figsize=(7,7))
            plt.title(txt)
            plt.imshow(image)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
                
            batch = []
            