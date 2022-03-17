import os.path

import numpy as np
import pandas as pd


def make_dict(data_csv_path, dict_id_path):
    train_csv = pd.read_csv(data_csv_path)
    train_csv_id = train_csv['Id'].unique()
    dict_id = dict(zip(train_csv_id, range(len(train_csv_id))))
    np.save(os.path.join(dict_id_path, "dict_id"), dict_id)
