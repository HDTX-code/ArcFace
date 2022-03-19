import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.Canny import get_img


def get_pre(test_path, model, Feature_train, target_train, dict_id, opt, it, device):
    Feature_train = torch.from_numpy(Feature_train).to(device)
    target_train = torch.from_numpy(target_train).to(device)

    model.eval()
    model.to(device)

    image = get_img(opt.th1, opt.th2, test_path, opt)
    test_tensor = torch.from_numpy(image)
    new_d = {v: k for k, v in dict_id.items()}
    num = len(dict_id)

    with torch.no_grad():
        test_tensor = test_tensor.type(torch.FloatTensor).to(device)
        feature_test = model(test_tensor.reshape(-1, 3, opt.h, opt.w).to(device)).to(device)
        output = F.cosine_similarity(
            torch.mul(torch.ones(Feature_train.shape).to(device), feature_test[0, :].T),
            Feature_train, dim=1).to(device)
        kind = torch.zeros([num]).to(device)
        for j in range(num):
            kind[j] = output[target_train[:, 0] == j].mean().to(device)
        sorted, indices = torch.sort(kind, descending=True)
        sorted = sorted.cpu().detach().numpy()
        indices = indices.cpu().detach().numpy()
        Top = sorted[:it]
        Top_index = []
        for item in range(it):
            Top_index.append(new_d[indices[item]])
        return Top, Top_index
