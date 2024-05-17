# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import os


class DataLogger:
    def __init__(self,
                 columns=["epoch","step", "error"],
                 save_path="modelCheckpoints"):
        self.data = []
        self.columns = columns
        self.save_path = Path(save_path)
        
    def add(self, x):
        self.data.append(x)
        
    def toDf(self):
        df = pd.DataFrame(self.data, columns=self.columns)
        return df
        
    def save(self):
        self.toDf().to_parquet(self.save_path/"loss.csv", index=False)
        
    def log(self, window=10):
        df = self.toDf()
        if df.shape[0] < window:
            print(df)
        else:
            print(df.rolling(window=window).mean().dropna())