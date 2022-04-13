import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def make_val(Feature_train, target_train, Feature_val, target_val, device, num):
    Feature_train = Feature_train.to(device)
    target_train = target_train.to(device)
    Feature_val = Feature_val.to(device)
    target_val = target_val.to(device)
    with torch.no_grad():
        Feature_train_num = np.zeros([num, 512])
        for item in range(num):
            Feature_train_num[item] = np.mean(Feature_train[target_train[:, 0] == item, :], axis=0)
        Feature_train_num = torch.from_numpy(Feature_train_num)
        with tqdm(total=len(target_val), postfix=dict) as pbar2:
            Score = 0
            for item in range(len(target_val)):
                output = F.cosine_similarity(
                    torch.mul(torch.ones(Feature_train_num.shape).to(device), Feature_val[item, :].T),
                    Feature_train_num, dim=1).to(device)
                # kind = torch.zeros([num]).to(device)
                # for j in range(num):
                #     kind[j] = output[target_train[:, 0] == j].mean().to(device)
                # sorted, indices = torch.sort(kind, descending=True)
                indices = output.argmax().to(device)
                if indices == target_val[item]:
                    Score = Score + 1
                pbar2.update(1)
                pbar2.set_postfix(**{'val_Score': Score / (item + 1)})
        #             if item>100:
        #                 break
    return Score / len(target_val)
