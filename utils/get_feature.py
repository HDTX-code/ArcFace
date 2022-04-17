import copy
import sys

import numpy as np
import torch
from tqdm import tqdm


def get_feature(model, dataloader, device, feature_num, get_id=False):
    with tqdm(total=len(dataloader)) as pbar3:
        with torch.no_grad():
            model.eval()
            model.to(device)
            val = 0
            if get_id:
                for iteration, (image_tensor, target_t, img_id) in enumerate(dataloader):
                    image_tensor = image_tensor.type(torch.FloatTensor).to(device)
                    feature = model(image_tensor.to(device))
                    feature.reshape(-1, feature_num).to(device)
                    target_t.reshape(-1, 1).to(device)
                    img_id.reshape(-1, 1).to(device)
                    if val == 0:
                        Feature = copy.copy(feature)
                        target = copy.copy(target_t)
                        Img_id = copy.copy(img_id)
                        val = 1
                    else:
                        Feature = torch.cat((Feature, feature), 0)
                        target = torch.cat((target, target_t), 0)
                        Img_id = torch.cat((Img_id, img_id), 0)
                    pbar3.update(1)
                return Feature, target, Img_id
            else:
                for iteration, (image_tensor, target_t) in enumerate(dataloader):
                    image_tensor = image_tensor.type(torch.FloatTensor).to(device)
                    feature = model(image_tensor.to(device))
                    feature.reshape(-1, feature_num).to(device)
                    target_t.reshape(-1, 1).to(device)
                    if val == 0:
                        Feature = copy.copy(feature)
                        target = copy.copy(target_t)
                        val = 1
                    else:
                        Feature = torch.cat((Feature, feature), 0)
                        target = torch.cat((target, target_t), 0)
                    pbar3.update(1)
                return Feature, target

