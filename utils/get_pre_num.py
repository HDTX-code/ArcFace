import numpy as np
import torch
import torch.nn.functional as F


def get_pre_num(feature_test, Feature_train_num, dict_id, dict_id_all, it, device):
    Feature_train_num = torch.from_numpy(Feature_train_num).to(device)
    feature_test = feature_test.to(device)
    new_d = {v: k for k, v in dict_id.items()}

    with torch.no_grad():
        output = F.cosine_similarity(
            torch.mul(torch.ones(Feature_train_num.shape).to(device), feature_test.T),
            Feature_train_num, dim=1).to(device)
        output = output.cpu().detach().numpy()
        Top = np.max(output)
        Top_index = dict_id_all[new_d[np.argmax(output)]]
        # sorted, indices = torch.sort(output, descending=True)
        # sorted = sorted.cpu().detach().numpy()
        # indices = indices.cpu().detach().numpy()
        # Top = sorted[:it]
        # Top_index = indices[:it]
        # for item3 in range(it):
        #     Top_index[item3] = dict_id_all[new_d[Top_index[item3]]]
        return Top, Top_index
