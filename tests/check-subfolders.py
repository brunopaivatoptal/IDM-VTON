# -*- coding: utf-8 -*-
"""
Check subfolders in the root folder
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
        [Path(r"E:\backups\toptal\pixelcut\virtual-try-on\viton_combined_annotated\viton_combined_annotated"),
         Path("/mnt/vdb/datasets/viton_combined_annotated/viton_combined_annotated")]
        if os.path.exists(x)][0]

results = {}
for folder in tqdm(os.listdir(root)):
    results.update({folder:os.listdir(root/folder)})
    
results_s = pd.Series(results)

results_s.apply(lambda x: len(x)).value_counts()