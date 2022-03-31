import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from utils.utils import get_img_for_tensor


class ClassesDataset(Dataset):
    def __init__(self, csv, w, h, is_strength=1):
        self.csv = csv
        self.w = w
        self.h = h
        self.is_strength = is_strength

    def __getitem__(self, index):
        path = self.csv.loc[index, 'path']
        target = self.csv.loc[index, 'individual_id']
        img1 = get_img_for_tensor(path, self.w, self.h)
        img_tensor = torch.from_numpy(img1)
        target_tensor = torch.ones([1])
        target_tensor[0] = target
        return img_tensor, target_tensor

    def __len__(self):
        return len(self.csv)
