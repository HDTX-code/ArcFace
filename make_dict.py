from __init__ import *


def make_dict(a, data_csv_path, dict_id_path):
    train_csv = pd.read_csv(data_csv_path)
    train_csv_id = train_csv['Id'].unique()
    dict_id = dict(zip(train_csv_id, range(len(train_csv_id))))
    info_json = json.dumps(dict_id, sort_keys=False, indent=4, separators=(',', ': '))
    f = open(os.path.join(dict_id_path, "dict_id"), 'w')
    f.write(info_json)


if __name__ == '__main__':
    make_dict(sys.argv[0], sys.argv[1], sys.argv[2])
