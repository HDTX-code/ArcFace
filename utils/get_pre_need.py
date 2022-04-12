import json
import os

import numpy as np
import pandas as pd
import timm
import torch
import torchvision
from torch.utils.data import DataLoader

from dataset.dataset import ArcDataset
from utils.get_feature import get_feature
from utils.utils import get_model


def get_pre_need(model_path, dict_id_path, dict_id_species_path, train_csv_path,  device, w, h, data_train_path,
                 batch_size, num_workers, save_path, backbone='resnet50', Feature_train_path=None, target_train_path=None):
    with torch.no_grad():
        pretrained = False
        # 拼接地址
        # model_path_Sph = os.path.join(root_path, backbone + "Sph.pth")
        # model_path_Arc = os.path.join(root_path, backbone + "Arc.pth")
        # model_path_Add = os.path.join(root_path, backbone + "Add.pth")
        # Feature_train_path = os.path.join(root_path, "Feature_train.npy")
        # target_train_path = os.path.join(root_path, "target_train.npy")
        # train_csv_train_path = os.path.join(root_path, "train_csv_train.csv")
        # dict_id_path = os.path.join(root_path, "dict_id")
        # if not os.path.exists(dict_id_path):
        #     dict_id_path = os.path.join(root_path, "dict_id.txt")

        # 加载模型
        model = get_model(backbone, pretrained, num_classes=512+ 30)

        model.load_state_dict(torch.load(model_path, map_location=device), False)
        model.eval()

        # 加载字典
        f2 = open(dict_id_path, 'r')
        dict_id = json.load(f2)

        f3 = open(dict_id_species_path, 'r')
        dict_id_species = json.load(f3)

        # 加载Feature_train
        if Feature_train_path is None:
            train_csv_train = pd.read_csv(train_csv_path)
            train_dataset = ArcDataset(train_csv_train, dict_id,
                                       dict_id_species, data_train_path, w, h)
            dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers)
            Feature_train, target_train = get_feature(model, dataloader, device, 512 + 30)
            Feature_train = Feature_train.cpu().detach().numpy()
            target_train = target_train.cpu().detach().numpy()
            np.save(os.path.join(save_path, "Feature_train.npy"), Feature_train)
            np.save(os.path.join(save_path, "target_train.npy"), target_train)
        else:
            Feature_train = np.load(Feature_train_path)
            target_train = np.load(target_train_path)

    return model, dict_id, dict_id_species, Feature_train, target_train
