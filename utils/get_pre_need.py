import json
import os

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from config.config import Config
from dataset.dataset import ArcDataset
from utils.get_feature import get_feature
from utils.make_csv import make_csv


def get_pre_need(root_path, device):
    with torch.no_grad():
        # 拼接地址
        model_path = os.path.join(root_path, "resnet50Sph.pth")
        Feature_train_path = os.path.join(root_path, "Feature_train.npy")
        target_train_path = os.path.join(root_path, "target_train.npy")
        dict_id_path = os.path.join(root_path, "dict_id.npy")

        # 加载模型
        model_Sph = torchvision.models.resnet50(pretrained=False)
        model_Sph.fc = torch.nn.Linear(model_Sph.fc.in_features, 512)
        model_Sph.load_state_dict(torch.load(model_path, map_location=device), False)
        model_Sph.eval()

        # 加载字典
        f2 = open(dict_id_path, 'r')
        dict_id = json.load(f2)

        # 加载Feature_train
        Feature_train = np.load(Feature_train_path)
        target_train = np.load(target_train_path)

    return model_Sph, dict_id, Feature_train, target_train
