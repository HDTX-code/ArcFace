import copy
import os
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.DataStrength import ImageNew


def get_csv(path, label):
    dir_list = os.listdir(path)
    csv = pd.DataFrame(columns=['image', 'path', 'individual_id'])
    for item in dir_list:
        csv.loc[len(csv)] = [item, os.path.join(path, item), label]
    return csv


def get_img_for_tensor(path, w, h, isNew=False):
    img = cv2.imread(path)
    if isNew:
        img = ImageNew(img)
    # 改变格式成规定的框和高
    if img.shape[2] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_0 = cv2.resize(img, (w, h))
    # 改变img1的时候不改变img
    img1 = np.zeros([3, w, h])
    # cv2读取的是bgr,转换成rgb就要做一下变通
    img1[0, :, :] = img_0[:, :, 2]
    img1[1, :, :] = img_0[:, :, 1]
    img1[2, :, :] = img_0[:, :, 0]
    return img1


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ---------------------------------------------------#
#   KNN
# ---------------------------------------------------#
def cal_distance(Feature_train, Feature_test, device, K):
    Feature_train = torch.from_numpy(Feature_train).to(device)
    Feature_test = Feature_test.to(device)
    with torch.no_grad():
        with tqdm(total=Feature_test.shape[0]) as pbar:
            val = 0
            for item in range(Feature_test.shape[0]):
                output = F.cosine_similarity(
                    torch.mul(torch.ones(Feature_train.shape).to(device), Feature_test[item, :].T),
                    Feature_train, dim=1).to(device)
                sorted, indices = torch.sort(output, descending=True)
                sorted = sorted.reshape(1, -1)
                indices = indices.reshape(1, -1)
                # output = output.reshape(1, -1)
                if val == 0:
                    Output_score = copy.copy(sorted[:K])
                    Output_index = copy.copy(indices[:K])
                    val = 1
                else:
                    Output_score = torch.cat((Output_score, sorted[:K]), 0)
                    Output_index = torch.cat((Output_index, indices[:K]), 0)
                pbar.update(1)
    return Output_score, Output_index


def KNN_by_iter(Feature_train, target_train, Feature_test, target_test, k, device,
                submission, new_d_test, new_id, save_path, Score_path=None, Index_path=None):
    # 计算距离
    # res = []
    if Score_path is None:
        print("计算数据")
        Score, Index = cal_distance(Feature_train, Feature_test, device, K=10000)
        Score = Score.cpu().detach().numpy()
        Index = Index.cpu().detach().numpy()
        np.save(os.path.join(save_path, "Score.npy"), Score)
        np.save(os.path.join(save_path, "Index.npy"), Index)
    else:
        print("加载数据")
        Score = np.load(Score_path)
        Index = np.load(Index_path)
    with tqdm(total=Feature_test.shape[0]) as pbar:
        for item in range(Feature_test.shape[0]):
            K = copy.copy(k)
            while True:
                print(Index[item, :K])
                print(Index[item, :K].dtype)
                target_train_index = target_train[Index[item, :K], 0]
                if len(np.unique(target_train_index)) >= 5:
                    break
                K += 5
            score = Score[item, :K]
            index = np.unique(target_train_index)
            res = np.zeros(len(index))
            for item2 in range(len(index)):
                res[item2] = score[target_train_index == index[item2]].mean()
            res_sort = res.argsort()[-5:]
            submission.loc[
                submission[
                    submission.image == new_d_test[target_test[item, 0]]].index.tolist(), "predictions"] = \
                new_id[Index[res_sort[-1]]] + ' ' + new_id[Index[res_sort[-2]]] + ' ' + new_id[
                    Index[res_sort[-3]]] + ' ' \
                + new_id[Index[res_sort[-4]]] + ' ' + new_id[Index[res_sort[-5]]]
            pbar.update(1)
    submission.to_csv(os.path.join(save_path, "submission.csv"), index=False)
