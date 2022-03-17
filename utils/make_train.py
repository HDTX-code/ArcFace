import os.path
import sys

import numpy as np
import torch
from tqdm import tqdm

from utils.get_feature import get_feature
from utils.make_val import make_val
from utils.save_model import save_model


def make_train(model, metric_fc, criterion, optimizer, scheduler, train_loader, val_loader,
               opt, device, num, Str):
    with tqdm(total=opt.max_epoch * (len(train_loader)), postfix=dict) as pbar:
        for i in range(opt.max_epoch):
            # 开始训练
            model = model.train()
            model.to(device)

            Loss = 0

            # 训练
            for iteration, (image_tensor, target_t) in enumerate(train_loader):
                image_tensor = image_tensor.type(torch.FloatTensor).to(device)
                feature = model(image_tensor).to(device)
                output = metric_fc(feature, target_t).to(device)
                loss = criterion(output.reshape(-1, opt.num_classes).to(device),
                                 target_t.reshape(-1).long().to(device)).to(device)
                Loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(**{'loss_{}'.format(Str): loss.item()})
                pbar.update(1)
                # if iteration >= 1:
                #     break
            scheduler.step()

            with torch.no_grad():
                model = model.eval()
                model.to(device)

                metric_fc = metric_fc.eval()
                metric_fc.to(device)

                Loss = Loss.cpu().detach().numpy() / len(train_loader)

                print("第{}轮 : Loss_{} = {}".format(i, Str, Loss))

                if i % opt.save_interval == 0 or i == opt.max_epoch - 1:
                    # 开始验证，获取特征矩阵
                    Feature_train, target_train = get_feature(model, train_loader, device)
                    Feature_val, target_val = get_feature(model, val_loader, device)
                    # 计算验证得分
                    Score = make_val(Feature_train, target_train, Feature_val, target_val, device, num)
                    path_model = os.path.join(opt.checkpoints_path, "model")
                    if not os.path.exists(path_model):
                        os.mkdir(path_model)
                    save_model(model, path_model, str(opt.backbone) + Str, i, Loss, Score)
                    path_featureMap = os.path.join(opt.checkpoints_path, "FeatureMap")
                    if not os.path.exists(path_featureMap):
                        os.mkdir(path_featureMap)
                    Feature_train = Feature_train.cpu().detach().numpy()
                    target_train = target_train.cpu().detach().numpy()
                    Feature_val = Feature_val.cpu().detach().numpy()
                    target_val = target_val.cpu().detach().numpy()
                    np.save(os.path.join(path_featureMap, "Feature_train_{}.npy".format(i)), Feature_train)
                    np.save(os.path.join(path_featureMap, "target_train_{}.npy".format(i)), target_train)
                    np.save(os.path.join(path_featureMap, "Feature_val_{}.npy".format(i)), Feature_val)
                    np.save(os.path.join(path_featureMap, "target_val_{}.npy".format(i)), target_val)
                    print("第{}轮 : Score={}".format(i, Score))
            # if i >= 1:
            #     break
