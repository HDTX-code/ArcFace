import json
import os

import pandas as pd


if __name__ == '__main__':
    # -------------------------------#
    #   参数设置
    # -------------------------------#
    data_csv_path = r"D:\project\happyWhale\efficentnet\train_csv_train.csv"
    pic_file = r"D:\project\happyWhale\classes\粗分类\result\no_head"
    save_path = r"C:\Users\12529\Desktop\新建文件夹"
    val_number = 0
    # -------------------------------#
    #   参数设置
    # -------------------------------#
    train_csv = pd.read_csv(data_csv_path)

    name_list = os.listdir(pic_file)
    train_csv_all = train_csv.loc[train_csv['image'].isin(name_list), ['image', 'individual_id']]
    train_csv_all.index = range(len(train_csv_all))
    train_csv_all_id = train_csv_all['individual_id'].unique()

    dict_id_all = dict(zip(train_csv_all_id, range(len(train_csv_all_id))))
    info_json = json.dumps(dict_id_all, sort_keys=False, indent=4, separators=(',', ': '))
    f = open(os.path.join(save_path, "dict_id"), 'w')
    f.write(info_json)

    train_csv_all.to_csv(os.path.join(save_path, "train_csv_train.csv"), index=False)

