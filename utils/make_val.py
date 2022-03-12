import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm


def make_val(Feature_train, target_train, Feature_val, target_val, device, num):
    Feature_train = torch.from_numpy(Feature_train).to(device)
    target_train = torch.from_numpy(target_train).to(device)
    Feature_val = torch.from_numpy(Feature_val).to(device)
    target_val = torch.from_numpy(target_val).to(device)
    with tqdm(total=len(target_val), postfix=dict, file=sys.stdout) as pbar:
        Score = 0
        for item in range(len(target_val)):
            output = F.cosine_similarity(
                torch.mul(torch.ones(Feature_train.shape).to(device), Feature_val[item, :].T.to(device)),
                Feature_train.to(device), dim=1).to(device)
            kind = torch.zeros(num)
            for j in range(num):
                kind[j] = sum(torch.where(target_train.reshape(-1).to(device) == j * torch.ones(target_train.shape).reshape(-1).to(device), output.reshape(-1).to(device),
                                          torch.zeros(output.shape).reshape(-1).to(device)).to(device)).to(device) / sum(target_train.reshape(-1).to(device) == j * torch.ones(target_train.shape).reshape(-1).to(device))
            sorted, indices = torch.sort(kind, descending=True)
            if sum(indices[:5].to(device) == target_val[item].to(device)).to(device) != 0:
                Score = Score + 1
            pbar.update(1)
            pbar.set_postfix(**{'val_Score': Score / (item + 1)})
    #             if item>100:
    #                 break
    return Score / len(target_val)
