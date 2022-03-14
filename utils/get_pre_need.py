import os

import torch
import torchvision

from config.config import Config
from utils.get_feature import get_feature
from utils.make_csv import make_csv


def get_pre_need(data_root_path, save_root_path, model_url, low, high, val_number, device):
    opt = Config()
    opt.data_train_path = os.path.join(data_root_path, "train")
    opt.data_csv_path = os.path.join(data_root_path, "train.csv")
    opt.data_test_path = os.path.join(data_root_path, "test")
    opt.checkpoints_path = save_root_path
    opt.low = int(low)
    opt.high = int(high)
    opt.val_number = int(val_number)

    model = torchvision.models.resnet50(pretrained=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 512)
    model.load_state_dict(torch.load(model_url, map_location=device), False)

    train_csv_train, train_csv_val, dict_id_all, new_d_all = make_csv(opt)

    Feature_train, target_train = get_feature(model, train_csv_train, device)

    return dict_id_all, new_d_all, Feature_train, target_train, opt, model
