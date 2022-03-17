import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm


def make_val(Feature_train, target_train, Feature_val, target_val, device, num):
    Feature_train = Feature_train.to(device)
    target_train = target_train.to(device)
    Feature_val = Feature_val.to(device)
    target_val = target_val.to(device)
    num = torch.from_numpy(num).to(device)
    num = num.to(device)
    with torch.no_grad():
        with tqdm(total=len(target_val), postfix=dict, file=sys.stdout) as pbar2:
            Score = 0
            for item in range(len(target_val)):
                output = F.cosine_similarity(
                    torch.mul(torch.ones(Feature_train.shape).to(device), Feature_val[item, :].T),
                    Feature_train, dim=1).to(device)
                kind = torch.zeros(num.shape)
                for j in range(len(num)):
                    kind[j] = output[target_train[:, 0] == num[j]].mean()
                sorted, indices = torch.sort(kind, descending=True)
                if sum(num[indices[:5].to(device)] == target_val[item]) != 0:
                    Score = Score + 1
                pbar2.update(1)
                pbar2.set_postfix(**{'val_Score': Score / (item + 1)})
        #             if item>100:
        #                 break
    return Score / len(target_val)
