import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.Canny import get_img


def get_pre(test_path, dict_id, dict_id_all, new_d_all, Feature_train, target_train, opt, model, device, th, it=-4):
    Feature_train = torch.from_numpy(Feature_train).to(device)
    target_train = torch.from_numpy(target_train).to(device)

    model.eval()
    model.to(device)

    image = get_img(opt.th1, opt.th2, test_path, opt)
    test_tensor = torch.from_numpy(image)
    Score = np.zeros(len(dict_id_all))

    with torch.no_grad():
        test_tensor = test_tensor.type(torch.FloatTensor).to(device)
        feature_test = model(test_tensor.reshape(-1, 3, opt.h, opt.w).to(device)).to(device)
        for j in range(len(dict_id_all)):
            Feature_train_F = Feature_train[(target_train == j).reshape(-1), :].to(device)
            output = F.cosine_similarity(
                torch.mul(torch.ones(Feature_train_F.shape).to(device), feature_test[0, :].T.to(device)),
                Feature_train_F.to(device), dim=1).to(device)
            score = output.cpu().detach().numpy()
            Score[j] = sum(score) / len(score)
        top4_index = Score.argsort()[it:]
        top4 = Score.sort()[it:]
        top4 = top4[top4 > th]
        top4_index = top4_index[top4 > th]
        f1 = np.frompyfunc(lambda x: dict_id(new_d_all(x)), 1, 1)
        top4_index = f1(top4_index)
        return top4, top4_index
