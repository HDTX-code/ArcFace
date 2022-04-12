import numpy as np
import torch
import torch.nn.functional as F


def get_pre_num(data_csv, feature_test, Feature_train_num, dict_id, dict_id_species, dict_id_all, it, device):
    feature_test = feature_test.cpu().detach().numpy()
    new_d = {v: k for k, v in dict_id.items()}
    new_d_species = {v: k for k, v in dict_id_species.items()}

    feature_species = feature_test[512:]
    species = new_d_species[np.argmax(feature_species)]
    Id_list = data_csv.loc[data_csv["species"] == species, "individual_id"].unique()
    id_index = []
    for item2 in Id_list:
        id_index.append(dict_id[item2])

    It = len(id_index) if len(id_index) < 5 else 5
    Top = np.zeros(5)
    Top_index = np.zeros(5)

    feature_train_num = Feature_train_num[id_index, :]

    feature_train_num = torch.from_numpy(feature_train_num).to(device)
    feature_test = torch.from_numpy(feature_test).to(device)
    with torch.no_grad():
        output = F.cosine_similarity(
            torch.mul(torch.ones(feature_train_num.shape).to(device), feature_test.T),
            feature_train_num, dim=1).to(device)
        output = output.cpu().detach().numpy()
        Top_index[:It] = id_index[np.argsort(output)[-It:][::-1]]
        Top[:It] = output[np.argsort(output)[-It:][::-1]]
        if len(Top) < 5:
            Top[It:5] = Top[It] * np.ones(5-It)
            Top_index[It:5] = Top_index[It] * np.ones(5 - It)
        # sorted, indices = torch.sort(output, descending=True)
        # sorted = sorted.cpu().detach().numpy()
        # indices = indices.cpu().detach().numpy()
        # Top = sorted[:it]
        # Top_index = indices[:it]
        for item in range(it):
            Top_index[item] = dict_id_all[new_d[Top_index[item]]]
        return Top, Top_index
