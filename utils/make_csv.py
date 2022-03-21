import copy
import json
import os

import pandas as pd

from config.config import Config


def make_csv(opt, dict_id_path):
    train_csv = pd.read_csv(opt.data_csv_path)
    train_csv_id = train_csv['Id'].unique()
    dict_id = dict(zip(train_csv_id, range(len(train_csv_id))))
    # 选取数据量大于等于 low 小于等于 high 的数据
    train_csv_describe = pd.DataFrame(columns=['Id', 'num'])
    train_csv_val = pd.DataFrame(columns=['Image', 'Id', 'd'])
    # 生成类别数量统计表
    for k, v in dict_id.items():
        train_csv_describe.loc[v] = [k, sum(train_csv['Id'] == k)]
    train_csv_describe = train_csv_describe.sort_values(by="num", ascending=False)
    train_csv_all_id = train_csv_describe.loc[train_csv_describe['num'].isin(range(opt.low, opt.high + 1)), 'Id']
    train_csv_all_id.index = range(len(train_csv_all_id))

    train_csv_all = train_csv.loc[train_csv['Id'].isin(train_csv_all_id), :]
    train_csv_all.index = range(len(train_csv_all))

    train_csv_all_1 = copy.copy(train_csv_all)

    train_csv_all.loc[:, 'd'] = 0
    # train_csv_val.sort_values(by="Image", inplace=True, ascending=True)
    # train_csv_val.index = range(len(train_csv_val))
    # train_csv_all.sort_values(by="Image", inplace=True, ascending=True)

    train_csv_all_0 = copy.copy(train_csv_all)

    for item in range(6, 12):
        F = copy.copy(train_csv_all_0)
        F.loc[:, 'd'] = item
        train_csv_all = pd.concat([train_csv_all, F], ignore_index=True)

    train_csv_all.index = range(len(train_csv_all))

    train_csv_train = copy.copy(train_csv_all)
    train_csv_val.loc[:, 'd'] = 0
    for item in train_csv_all_id:
        train_csv_val = pd.concat([train_csv_val, train_csv_all.loc[
                                                  train_csv_all[(train_csv_all["Id"] == item) & (train_csv_all["d"] == 0)].index.tolist()[
                                                  :opt.val_number], :]], ignore_index=True)
        train_csv_train = train_csv_train[~train_csv_train.loc[:, :].isin(train_csv_train.loc[
                                                  train_csv_train[(train_csv_all["Id"] == item) & (train_csv_all["d"] == 0)].index.tolist()[
                                                  :opt.val_number], :])]
        train_csv_train = train_csv_train.dropna(axis=0, how='any')


    train_csv_train.index = range(len(train_csv_train))
    train_csv_val.index = range(len(train_csv_val))

    dict_id_all = dict(zip(train_csv_all_id, range(len(train_csv_all_id))))
    info_json = json.dumps(dict_id_all, sort_keys=False, indent=4, separators=(',', ': '))
    f = open(os.path.join(dict_id_path, "dict_id"), 'w')
    f.write(info_json)

    train_csv_train.to_csv(os.path.join(opt.checkpoints_path, "train_csv_train.csv"), index=False)
    train_csv_val.to_csv(os.path.join(opt.checkpoints_path, "train_csv_val.csv"), index=False)
    return train_csv_train, train_csv_val, dict_id_all


#
if __name__ == '__main__':
    opt = Config()
    train_csv_train, train_csv_val, dict_id_all = make_csv(opt,
                                                           r"D:\project\humpWhale\arcFace\ArcFace-modification-\data")
    print(train_csv_val.head())
