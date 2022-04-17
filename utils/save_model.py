import os

import torch


def save_model(model, save_path, name, iter_cnt, loss):
    save_name = os.path.join(save_path,
                             name + '-' + str(iter_cnt) + 'loss: ' + str(loss) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name
