import pandas as pd


def make_csv(opt):
    train_csv = pd.read_csv(opt.data_csv_path)
    train_csv_id = train_csv['Id'].unique()
    dict_id = dict(zip(train_csv_id, range(len(train_csv_id))))

    # 选取数据量大于 10 小于 30 的数据
    train_csv_describe = pd.DataFrame(columns=['Id', 'num'])
    train_csv_train = pd.DataFrame(columns=['Image', 'Id'])
    train_csv_val = pd.DataFrame(columns=['Image', 'Id'])
    # 生成类别数量统计表
    for k, v in dict_id.items():
        train_csv_describe.loc[v] = [k, sum(train_csv['Id'] == k)]
    train_csv_describe = train_csv_describe.sort_values(by="num", ascending=False)
    train_csv_all_id = train_csv_describe.loc[train_csv_describe['num'].isin(range(opt.low, opt.high + 1)), 'Id']
    train_csv_all_id.index = range(len(train_csv_all_id))

    train_csv_all = train_csv.loc[train_csv['Id'].isin(train_csv_all_id), :]
    train_csv_all.index = range(len(train_csv_all))

    for item in train_csv_all_id:
        if opt.val_number == 0:
            train_csv_train = pd.concat([train_csv_train, train_csv_all.loc[train_csv_all[train_csv_all["Id"] == item].index.tolist(), :]], ignore_index=True)
        else:
            train_csv_train = pd.concat([train_csv_train, train_csv_all.loc[train_csv_all[train_csv_all["Id"] == item].index.tolist()[opt.val_number:], :]], ignore_index=True)
            train_csv_val = pd.concat([train_csv_val, train_csv_all.loc[train_csv_all[train_csv_all["Id"] == item].index.tolist()[:opt.val_number], :]], ignore_index=True)
    dict_id_all = dict(zip(train_csv_all_id, range(len(train_csv_all_id))))
    new_d_all = {v: k for k, v in dict_id_all.items()}

    return train_csv_train, train_csv_val, dict_id_all, new_d_all
