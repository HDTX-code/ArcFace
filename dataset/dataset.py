import os
import torch
from torch.utils.data import Dataset
from utils.Canny import get_img
from utils.DataStrength import Data_strength


class ArcDataset(Dataset):
    def __init__(self, opt, csv, dict_id):
        self.opt = opt
        self.csv = csv
        self.dict_id = dict_id

    def __getitem__(self, index):
        path = os.path.join(self.opt.data_train_path, self.csv.loc[index, 'Image'])
        target = self.dict_id[self.csv.loc[index, 'Id']]
        num = self.csv.loc[index, 'd']
        if int(num) == 0:
            image = get_img(self.opt.th1, self.opt.th2, path, self.opt)
        else:
            image = Data_strength(path, int(num), self.opt)
        img_tensor = torch.from_numpy(image)
        target_tensor = torch.ones([1])
        target_tensor[0] = target
        return img_tensor, target_tensor

    def __len__(self):
        return len(self.csv)
