# -*- coding: utf-8 -*-
"""
Separate incomplete folders
"""
from pathlib import Path 
from tqdm import tqdm
import pandas as pd 
import numpy as np
import shutil
import os

trainig_data_possible_dirs : tuple =(
  r"E:\backups\toptal\pixelcut\virtual-try-on\viton_combined_annotated\viton_combined_annotated",
  "/mnt/vdb/datasets/viton_combined_annotated/viton_combined_annotated"
)

srcDir = Path([x for x in trainig_data_possible_dirs if os.path.exists(x)][0])
dstDir = srcDir/"../removed_samples"

os.makedirs(dstDir, exist_ok=True)

allFolders = os.listdir(srcDir)
numSubfolders = {f:len(os.listdir(srcDir/f)) for f in tqdm(allFolders, desc="Counting subfolders...")}

s_SubfolderCounts = pd.Series(numSubfolders)
s_SubfolderCounts.value_counts()

toMove = s_SubfolderCounts[s_SubfolderCounts < 5].copy()

for m in tqdm(toMove.index, desc="Moving..."):
    shutil.move(srcDir/m, dstDir/m)