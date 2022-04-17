import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def make_val(Feature_train, target_train, Feature_val, target_val, device, num, Img_id_val,
             new_id_all, new_val_id, train_csv_val):
    with torch.no_grad():
        # Feature_train = Feature_train.to(device)
        # target_train = target_train.to(device)
        Feature_val = Feature_val.to(device)
        # target_val = target_val.to(device)
        Feature_train_num = np.zeros([num, 512])
        for item in range(num):
            Feature_train_num[item] = np.mean(Feature_train[target_train[:, 0] == item, :], axis=0)
        Feature_train_num = torch.from_numpy(Feature_train_num).to(device)

        analyse_right = pd.DataFrame(columns=['image', 'species', 'individual_id', 'score',
                                              'predictions_1', 'predictions_2', 'predictions_3', 'predictions_4',
                                              'predictions_5'])
        analyse_error = pd.DataFrame(columns=['image', 'species', 'individual_id',
                                              'predictions_1', 'predictions_2', 'predictions_3', 'predictions_4',
                                              'predictions_5'])
        map_1, map_2, map_3, map_4, map_5 = 0, 0, 0, 0, 0
        for item in range(len(target_val)):
            with tqdm(total=len(target_val), postfix=dict) as pbar2:
                output = F.cosine_similarity(
                    torch.mul(torch.ones(Feature_train_num.shape).to(device), Feature_val[item, :].T),
                    Feature_train_num, dim=1).to(device)
                # kind = torch.zeros([num]).to(device)
                # for j in range(num):
                #     kind[j] = output[target_train[:, 0] == j].mean().to(device)
                # sorted, indices = torch.sort(kind, descending=True)
                sorted, indices = torch.sort(output, descending=True)
                sorted = sorted.cpu().detach().numpy()
                indices = indices.cpu().detach().numpy()
                if int(indices[0]) == int(target_val[item, 0]):
                    map_1 = map_1 + 1
                    score = 0
                elif int(indices[1]) == int(target_val[item, 0]):
                    map_2 = map_2 + 1
                    score = 1
                elif int(indices[2]) == int(target_val[item, 0]):
                    map_3 = map_3 + 1
                    score = 2
                elif int(indices[3]) == int(target_val[item, 0]):
                    map_4 = map_4 + 1
                    score = 3
                elif int(indices[4]) == int(target_val[item, 0]):
                    map_5 = map_5 + 1
                    score = 4
                else:
                    score = 5
                MAP5 = (1 / 5) * map_5 + (1 / 4) * map_4 + (1 / 3) * map_3 + (1 / 2) * map_2 + (1 / 1) * map_1
                pbar2.update(1)
                pbar2.set_postfix(**{'val_Score': (map_1 / (item + 1)) * 1000 // 1000,
                                     'MAP5': (MAP5 / (item + 1)) * 1000 // 1000})
                if score != 5:
                    analyse_right.loc[len(analyse_right), :] = [new_val_id[Img_id_val[item, 0]],
                                                                train_csv_val.loc[
                                                                    train_csv_val['image'] == new_val_id[
                                                                        Img_id_val[item, 0]],
                                                                    'species'].values[0],
                                                                train_csv_val.loc[
                                                                    train_csv_val['image'] == new_val_id[
                                                                        Img_id_val[item, 0]],
                                                                    'individual_id'].values[0],
                                                                score,
                                                                new_id_all[indices[0]] + '-' + str(sorted[0]),
                                                                new_id_all[indices[1]] + '-' + str(sorted[1]),
                                                                new_id_all[indices[2]] + '-' + str(sorted[2]),
                                                                new_id_all[indices[3]] + '-' + str(sorted[3]),
                                                                new_id_all[indices[4]] + '-' + str(sorted[4])]
                else:
                    analyse_error.loc[len(analyse_error), :] = [new_val_id[Img_id_val[item, 0]],
                                                                train_csv_val.loc[
                                                                    train_csv_val['image'] == new_val_id[
                                                                        Img_id_val[item, 0]],
                                                                    'species'].values[0],
                                                                train_csv_val.loc[
                                                                    train_csv_val['image'] == new_val_id[
                                                                        Img_id_val[item, 0]],
                                                                    'individual_id'].values[0],
                                                                new_id_all[indices[0]] + '-' + str(sorted[0]),
                                                                new_id_all[indices[1]] + '-' + str(sorted[1]),
                                                                new_id_all[indices[2]] + '-' + str(sorted[2]),
                                                                new_id_all[indices[3]] + '-' + str(sorted[3]),
                                                                new_id_all[indices[4]] + '-' + str(sorted[4])]
        print('val_Score: ' + str((map_1 / (len(target_val))) * 1000 // 1000))
        print('MAP5: ' + str((MAP5 / (len(target_val))) * 1000 // 1000))
        #             if item>100:
        #                 break
    return analyse_right, analyse_error
