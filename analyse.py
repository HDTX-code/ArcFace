from __init__ import *


def analyse(args):
    # 读取
    # 生成spices对应的字典与反字典
    train_csv = pd.read_csv(args.data_csv_path)

