# -*- coding: utf-8 -*-
from modules.DataLoading import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

ds = SDCNVTONDataset(data_dir=r"E:\backups\toptal\pixelcut\virtual-try-on\viton_combined_annotated\viton_combined_annotated",
                     pretrained_processor_path="openai/clip-vit-large-patch14")
self=ds
index=0

plt.figure(figsize=(12,10))
for i in range(12):
    plt.subplot(3,4,i+1)
    x = ds[np.random.randint(0, len(ds))]
    plt.imshow(x["cloth_mask"].cpu().numpy().transpose(1,2,0))
    plt.axis("off")
plt.tight_layout()
plt.show()