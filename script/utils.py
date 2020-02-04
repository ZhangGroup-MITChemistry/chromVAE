__author__ = "Wenjun Xie <xwj123@gmail.com>"

import numpy as np
from torch.utils.data import Dataset, DataLoader

class Binary_Dataset(Dataset):
    def __init__(self, seq):
        super(Binary_Dataset).__init__()
        self.seq = seq

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx):
        return self.seq[idx, :]
