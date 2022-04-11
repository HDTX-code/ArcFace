import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.Canny import get_img
from utils.DataStrength import Data_strength
from utils.utils import get_img_for_tensor


class ArcDataset(Dataset):
    def __init__(self, csv, dict_id, dict_id_species, data_train_path, w, h):
        self.data_train_path = data_train_path
        self.w = w
        self.h = h
        self.csv = csv
        self.dict_id = dict_id
        self.dict_id_species = dict_id_species

    def __getitem__(self, index):
        path = os.path.join(self.data_train_path, self.csv.loc[index, 'image'])
        target = self.dict_id[self.csv.loc[index, 'individual_id']]
        species = self.dict_id_species[self.csv.loc[index, 'species']]
        img1 = get_img_for_tensor(path, self.w, self.h)
        img_tensor = torch.from_numpy(img1)

        target_tensor = torch.ones([1])
        target_tensor[0] = target

        species_label = torch.ones([1])
        species_label[0] = species

        return img_tensor, target_tensor, species_label

    def __len__(self):
        return len(self.csv)
