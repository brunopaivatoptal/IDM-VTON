# -*- coding: utf-8 -*-
"""
This module contains a Dataset to be used with Pixelcut's Virtual Try On dataset

To test:

    self = VtonDataset(r"E:\backups\toptal\pixelcut\virtual-try-on\viton-partial", (1024, 1024))
    
"""
from modules.SegmentationLabels import CIHP_LABELS
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import pickle
import torch
import json
import os

def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    if len(images.shape) == 3:
        images = torch.from_numpy(images.transpose(2, 0, 1))
    elif len(images.shape) == 2:
        images = torch.from_numpy(images).unsqueeze(0)
    return images

class VtonDataset(Dataset):
    def __init__(self, root, imgSize):
        """
        Args:
            data (list): List of data items.
            labels (list): List of labels corresponding to the data items.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if type(root) is str:
            root = Path(root)
        self.root = root
        self.initializeIndex(root)
        self.imgSize = imgSize
        self.transform = T.Compose(
            [
                T.Normalize([0.5], [0.5]),
            ])
    
    def initializeIndex(self, root):
        idx = []
        for x in tqdm(os.listdir(root), desc="Initializing File Index..."):
            currIndex = root/x
            if not os.path.isdir(currIndex):
                continue
            files = {x: currIndex/x/os.listdir(currIndex/x)[0]
                     if len(os.listdir(currIndex/x))>0 
                     else None for x in os.listdir(currIndex)}
            idx.append(files)
        self.index = idx
        
    def loadFile(self, fn):
        try:
            ext = str(fn).split(".")[-1]
            if ext in set(["pickle", "pkl"]):
                with open(fn, "rb") as f:
                    data = pickle.load(f)
            elif ext in set(['txt']):
                with open(fn) as file:
                    data = file.read()
            else:
                data = pil_to_tensor(Image.open(fn)).unsqueeze(0)
                data = self.transform(data)
            return data
        except:
            return None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        item = self.index[index]
        dataObjs = {x:self.loadFile(item[x]) for x in item}
        dataObjs["caption_cloth"] = "a photo of " + dataObjs["caption"]
        dataObjs["caption"] = "model is wearing a " + dataObjs["caption"]
        dataObjs.update({"filename":os.path.split(item)[0]})
        return dataObjs
