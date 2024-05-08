# -*- coding: utf-8 -*-
"""
Captioning model:
    https://huggingface.co/docs/transformers/main/en/model_doc/idefics2
"""
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
from pathlib import Path
from PIL import Image
import pandas as pd 
import numpy as np 
import requests
import os 

root = [x for x in 
        [Path(r"E:\backups\toptal\pixelcut\virtual-try-on\viton-partial"),
         Path("/mnt/vdb/datasets/viton_combined_annotated/viton_combined_annotated")]
        if os.path.exists(x)][0]
BATCH_SZ=2

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b")

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What’s the difference between these two images?"},
        {"type": "image"},
        {"type": "image"},
    ],
}]

