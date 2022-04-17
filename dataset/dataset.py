import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.Canny import get_img
from utils.DataStrength import Data_strength
from utils.utils import get_img_for_tensor


class ArcDataset(Dataset):
    def __init__(self, csv, dict_id, data_train_path, w, h, label='individual_id', IsNew=False, dict_train_id=None):
        self.data_train_path = data_train_path
        self.w = w
        self.h = h
        self.csv = csv
        self.dict_id = dict_id
        self.dict_train_id = dict_train_id
        self.IsNew = IsNew
        self.label = label

    def __getitem__(self, index):
        path = os.path.join(self.data_train_path, self.csv.loc[index, 'image'])
        target = self.dict_id[self.csv.loc[index, self.label]]
        img1 = get_img_for_tensor(path, self.w, self.h, self.IsNew)
        img_tensor = torch.from_numpy(img1)
        target_tensor = torch.ones([1])
        target_tensor[0] = target
        if self.dict_train_id is None:
            return img_tensor, target_tensor
        else:
            img_id = torch.ones([1])
            id = self.dict_train_id[self.csv.loc[index, 'image']]
            img_id[0] = id
            return img_tensor, target_tensor, img_id

    def __len__(self):
        return len(self.csv)
