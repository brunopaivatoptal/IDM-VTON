# -*- coding: utf-8 -*-
"""
This module contains a Dataset to be used with Pixelcut's Virtual Try On dataset

To test:

    self = VtonDataset(r"E:\backups\toptal\pixelcut\virtual-try-on\viton-partial", (1024, 1024))
    
"""

from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import pickle
import torch
import json
import os

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
                T.ToTensor(),
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
            else:
                data = Image.open(fn)
            return data
        except:
            return None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        item = self.index[index]
        images = {x:self.loadFile(item[x]) for x in item}
        return item
