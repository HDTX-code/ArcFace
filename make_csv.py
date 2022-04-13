from __init__ import *


def get_dict_csv(args):
    train_csv = pd.read_csv(args.data_csv_path)

    name_list = os.listdir(args.data_train_path)
    train_csv_all = train_csv.loc[train_csv['image'].isin(name_list), ['image', 'species', 'individual_id']]
    train_csv_all.index = range(len(train_csv_all))
    train_csv_all_id = train_csv_all[args.label].unique()

    dict_id_all = dict(zip(train_csv_all_id, range(len(train_csv_all_id))))
    info_json = json.dumps(dict_id_all, sort_keys=False, indent=4, separators=(',', ': '))
    f = open(os.path.join(args.save_path, "dict_id"), 'w')
    f.write(info_json)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    train_csv_all.to_csv(os.path.join(args.save_path, "data_csv.csv"), index=False)

    data_csv = train_csv_all
    train_csv_train = pd.DataFrame(columns=['image', 'species', 'individual_id'])
    train_csv_val = pd.DataFrame(columns=['image', 'species', 'individual_id'])
    for item in data_csv[args.label].unique():
        if len(data_csv[data_csv[args.label] == item].index.tolist()) * args.val >= 1:
            It = math.floor(len(data_csv[data_csv[args.label] == item].index.tolist()) * args.val)
            print(item + ':' + str(It))
            train_csv_train = pd.concat([train_csv_train,
                                         data_csv.loc[data_csv[data_csv[args.label] == item].index.tolist()[It:],
                                         :]])
            train_csv_val = pd.concat([train_csv_val,
                                       data_csv.loc[data_csv[data_csv[args.label] == item].index.tolist()[:It],
                                       :]])

    train_csv_train.to_csv(
        os.path.join(args.save_path, "train_csv_train.csv"), index=False)
    if len(train_csv_val) > 0:
        train_csv_val.to_csv(
            os.path.join(args.save_path, "train_csv_val.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成训练所需dict、csv的参数设置')
    parser.add_argument('--data_csv_path', type=str, help='全体训练集csv路径',
                        default=r'../input/happy-whale-and-dolphin/train.csv')
    parser.add_argument('--save_path', type=str, help='存储路径', default=r'./')
    parser.add_argument('--label', type=str, help='标签列名', default='individual_id')
    parser.add_argument('--data_train_path', type=str, help='训练集路径', required=True)
    parser.add_argument('--val', type=float, help='验证集合占比', default=0.2)
    args = parser.parse_args()

    get_dict_csv(args)
