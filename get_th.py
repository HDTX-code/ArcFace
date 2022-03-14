from __init__ import *


def get_th(a, data_root_path, save_root_path, low, high, val_number, model_url):
    opt = Config()
    opt.data_train_path = os.path.join(data_root_path, "train")
    opt.data_csv_path = os.path.join(data_root_path, "train.csv")
    opt.data_test_path = os.path.join(data_root_path, "test")
    opt.checkpoints_path = save_root_path
    opt.low = int(low)
    opt.high = int(high)
    opt.val_number = int(val_number)
    # 训练设备
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 清洗数据，生成训练所需csv及dict
    train_csv_train, train_csv_val, dict_id_all, new_d_all = make_csv(opt)
    opt.num_classes = len(dict_id_all)
    # 生成train、val的dataloader
    train_dataset = ArcDataset(opt, train_csv_train, dict_id_all)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.num_workers)
    val_dataset = ArcDataset(opt, train_csv_val, dict_id_all)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=True,
                                num_workers=opt.num_workers)

    # 加载backbone
    model_Sph = torchvision.models.resnet50(pretrained=None)
    model_Sph.fc = torch.nn.Linear(model_Sph.fc.in_features, 512)
    model_Sph.load_state_dict(torch.load(model_url, map_location=device), False)

    model_Sph.to(device)
    model_Sph.eval()

    # 开始验证，获取特征矩阵
    Feature_train, target_train = get_feature(model_Sph, train_dataloader, device)
    Feature_val, target_val = get_feature(model_Sph, val_dataloader, device)

    # 计算阈值
    Feature_train = torch.from_numpy(Feature_train).to(device)
    target_train = torch.from_numpy(target_train).to(device)
    Feature_val = torch.from_numpy(Feature_val).to(device)
    target_val = torch.from_numpy(target_val).to(device)

    with tqdm(total=len(target_val), postfix=dict, file=sys.stdout) as pbar2:
        th = 10
        for item in range(len(target_val)):
            Feature_train_item = Feature_train[target_train == target_val[item], :]
            output = F.cosine_similarity(
                torch.mul(torch.ones(Feature_train_item.shape).to(device), Feature_val[item, :].T.to(device)),
                Feature_train_item.to(device), dim=1).to(device)
            kind = output.mean().to(device)
            if kind <= th:
                th = kind
        pbar2.set_postfix(**{'th:': th})
        pbar2.update(1)
    return th


if __name__ == '__main__':
    get_th(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
