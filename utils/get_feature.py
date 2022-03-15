import sys

import numpy as np
import torch
from tqdm import tqdm


def get_feature(model, dataloader, device):
    model.eval()
    model.to(device)
    val = 0
    with torch.no_grad():
        with tqdm(total=len(dataloader), file=sys.stdout) as pbar3:
            for iteration, (image_tensor, target_t) in enumerate(dataloader):
                image_tensor = image_tensor.type(torch.FloatTensor).to(device)
                feature = model(image_tensor.to(device))
                feature.reshape(-1, 512).to(device)
                target_t.reshape(-1, 1).to(device)
                feature = feature.cpu().detach().numpy()
                target_t = target_t.cpu().detach().numpy()
                if val == 0:
                    Feature = feature
                    target = target_t
                    val = 1
                else:
                    Feature = np.concatenate((Feature, feature), axis=0)
                    target = np.concatenate((target, target_t), axis=0)
                pbar3.update(1)
    return Feature, target
