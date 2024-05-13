# -*- coding: utf-8 -*-
"""
Aggregate all the generated captions 
"""
from pathlib import Path 
from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import os

def readFile(fn):
    with open(fn) as f:
        data = f.read()
    return data

root = [x for x in 
        [
            Path(r"E:\backups\toptal\pixelcut\virtual-try-on\viton_combined_annotated\viton_combined_annotated"),
            Path(r"E:\backups\toptal\pixelcut\virtual-try-on\viton_hd_images_test_paired_annotated_open_pose_yolo_pose"),
            Path("/mnt/vdb/datasets/viton_combined_annotated/viton_combined_annotated"),
         ]
        if os.path.exists(x)][0]
allSubFolders = os.listdir(root)

allCaptions = {}

for subfolder in tqdm(allSubFolders, desc="Aggregating captions..."):
    captionFn = root/subfolder/"caption/caption.txt"
    caption = readFile(captionFn)
    allCaptions.update({subfolder:caption})
    
allCaptions_s = pd.DataFrame(pd.Series(allCaptions, name="caption"))
allCaptions_s.to_pickle(root/"../../captions_viton_combined_annotated.pickle")

