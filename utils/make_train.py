import os.path
import sys

import numpy as np
import torch
from tqdm import tqdm

from utils.get_feature import get_feature
from utils.make_val import make_val
from utils.save_model import save_model
from utils.utils import get_lr


def make_train(model, metric_fc, criterion, optimizer, scheduler,
               train_loader, device, Str, num_classes, criterion_species,
               max_epoch, save_interval, save_path, backbone, epoch_start, epoch_end, Freeze_Epoch, val_loader=None):
    for item in range(epoch_start, epoch_end + 1):
        with tqdm(total=(len(train_loader)), desc=f'Epoch {item}/{max_epoch}', postfix=dict) as pbar:
            # 开始训练
            model = model.train()
            model.to(device)

            Loss_target = 0
            Loss_species = 0

            # 训练
            for iteration, (image_tensor, target_t, species_t) in enumerate(train_loader):
                image_tensor = image_tensor.type(torch.FloatTensor).to(device)

                feature = model(image_tensor).to(device)
                # print(feature.shape)
                feature_target = feature[:, :512].to(device)
                # print(feature_target.shape)
                feature_species = feature[:, 512:].to(device)
                # print(feature_species.shape)

                output_target = metric_fc(feature_target, target_t).to(device)
                loss_target = criterion(output_target.reshape(-1, num_classes).to(device),
                                        target_t.reshape(-1).long().to(device)).to(device)

                loss_species = criterion_species(feature_species.to(device), species_t.reshape(-1).long().to(device)).to(device)

                loss = loss_target * 0.3 + loss_species * 0.7
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                Loss_target += loss_target.cpu().detach().numpy()
                Loss_species += loss_species.cpu().detach().numpy()

                pbar.set_postfix(**{'{}'.format(Str): loss_target.item(),
                                    'species': loss_species.item(),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)
                # if iteration >= 1:
                #     break
            scheduler.step()

        with torch.no_grad():
            model = model.eval()
            model.to(device)

            metric_fc = metric_fc.eval()
            metric_fc.to(device)

            Loss_target = Loss_target / len(train_loader)
            Loss_species = Loss_species / len(train_loader)

            print("第{}轮 : Loss_{} = {}".format(item, Str, Loss_target))
            print("第{}轮 : Loss_{} = {}".format(item, 'species', Loss_species))

            if (item % save_interval == 0 or item == max_epoch) and item > Freeze_Epoch:
                # 开始验证，获取特征矩阵
                if item == max_epoch:
                    Feature_train, target_train = get_feature(model, train_loader, device, 512 + 30)
                    path_featureMap = os.path.join(save_path, "FeatureMap")
                    if not os.path.exists(path_featureMap):
                        os.mkdir(path_featureMap)
                    Feature_train = Feature_train.cpu().detach().numpy()
                    target_train = target_train.cpu().detach().numpy()
                    np.save(os.path.join(path_featureMap, "Feature_train_{}.npy".format(item)), Feature_train)
                    np.save(os.path.join(path_featureMap, "target_train_{}.npy".format(item)), target_train)
                if val_loader is not None:
                    # 计算验证得分
                    Feature_val, target_val = get_feature(model, val_loader, device, 512 + 30)
                    Score = make_val(Feature_train, target_train, Feature_val, target_val, device, num_classes)
                else:
                    Score = 0
                path_model = os.path.join(save_path, "model")
                if not os.path.exists(path_model):
                    os.mkdir(path_model)
                save_model(model, path_model, str(backbone) + Str, item, Loss, Score)
                # Feature_val = Feature_val.cpu().detach().numpy()
                # target_val = target_val.cpu().detach().numpy()
                # np.save(os.path.join(path_featureMap, "Feature_val_{}.npy".format(epoch_now)), Feature_val)
                # np.save(os.path.join(path_featureMap, "target_val_{}.npy".format(epoch_now)), target_val)
                # print("第{}轮 : Score={}".format(i, Score))
        # if i >= 1:
        #     break
    return model
