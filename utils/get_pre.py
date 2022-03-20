import sys

import cv2
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

    image = cv2.imread(test_path)
    img_0 = cv2.resize(image, (opt.w, opt.h))
    img1 = np.zeros([3, opt.w, opt.h])  # 改变img1的时候不改变img
    img1[0, :, :] = img_0[:, :, 2]
    img1[1, :, :] = img_0[:, :, 1]
    img1[2, :, :] = img_0[:, :, 0]  # cv2读取的是bgr,转换成rgb就要做一下变通
    test_tensor = torch.from_numpy(img1)
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
