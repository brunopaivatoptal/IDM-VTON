# -*- coding: utf-8 -*-
"""
Extracts a captions.pickle file into multiple .txt files
"""

from pathlib import Path 
from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import os

srcFolder = Path(r"E:\backups\toptal\pixelcut\virtual-try-on")
dstFolder = Path(r"E:\backups\toptal\pixelcut\virtual-try-on\viton_combined_annotated\viton_combined_annotated")
captionsFn = "viton_combined_annotated_captions.pickle"

captions = pd.read_pickle(srcFolder/captionsFn)

for f in tqdm(captions.index, desc="Extracting captions..."):
    caption = captions.loc[f].values[0]
    currFolder = dstFolder/f
    os.makedirs(currFolder/"caption", exist_ok=True)
    with open(currFolder/"caption"/"caption.txt", "w") as file:
        file.write(caption)
    