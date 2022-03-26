import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.Canny import get_img
from utils.DataStrength import Data_strength


class ArcDataset(Dataset):
    def __init__(self, csv, dict_id, data_train_path, w, h):
        self.data_train_path = data_train_path
        self.w = w
        self.h = h
        self.csv = csv
        self.dict_id = dict_id

    def __getitem__(self, index):
        path = os.path.join(self.data_train_path, self.csv.loc[index, 'image'])
        target = self.dict_id[self.csv.loc[index, 'Id']]
        img = cv2.imread(path)
        # 改变格式成规定的框和高
        if img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_0 = cv2.resize(img, (self.w, self.h))
        img1 = np.zeros([3, self.w, self.h])  # 改变img1的时候不改变img
        img1[0, :, :] = img_0[:, :, 2]
        img1[1, :, :] = img_0[:, :, 1]
        img1[2, :, :] = img_0[:, :, 0]  # cv2读取的是bgr,转换成rgb就要做一下变通
        img_tensor = torch.from_numpy(img1)
        target_tensor = torch.ones([1])
        target_tensor[0] = target
        return img_tensor, target_tensor

    def __len__(self):
        return len(self.csv)
