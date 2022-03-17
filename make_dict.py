from __init__ import *


def make_dict(a, data_csv_path, dict_id_path):
    train_csv = pd.read_csv(data_csv_path)
    train_csv_id = train_csv['Id'].unique()
    dict_id = dict(zip(train_csv_id, range(len(train_csv_id))))
    np.save(os.path.join(dict_id_path, "dict_id"), dict_id)


if __name__ == '__main__':
    make_dict(sys.argv[0], sys.argv[1], sys.argv[2])
