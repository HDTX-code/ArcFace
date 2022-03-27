import numpy as np
import torch
import torch.nn.functional as F


def get_feature_num(feature_num, device):
    with torch.no_grad:
        feature_num_R = np.zeros([feature_num.shape[0], feature_num.shape[0]])
        feature_num_tensor = torch.from_numpy(feature_num).to(device)
        for item in range(feature_num.shape[0]):
            output = F.cosine_similarity(
                torch.mul(torch.ones(feature_num_tensor.shape).to(device), feature_num_tensor[item, :].T),
                feature_num_tensor, dim=1).to(device)
            output = output.cpu().detach().numpy()
            feature_num_R[item, :] = output
        Weight_sum = np.sum(feature_num_R, axis=0)
        Weight = Weight_sum/sum(Weight_sum)
        Feature_num = np.matmul(Weight.T, feature_num)
        return Feature_num

