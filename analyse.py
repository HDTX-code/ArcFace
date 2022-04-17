from __init__ import *


def analyse(args):
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    f2 = open(args.dict_id_path, 'r')
    dict_id_all = json.load(f2)
    new_id_all = {v: k for k, v in dict_id_all.items()}

    train_csv_train = pd.read_csv(args.train_csv_train_path)

    train_csv_val = pd.read_csv(args.train_csv_val_path)
    dict_val_id = dict(zip(train_csv_val['image'].values, range(len(train_csv_val['image'].values))))
    info_json2 = json.dumps(dict_val_id, sort_keys=False, indent=4, separators=(',', ': '))
    f2 = open(os.path.join(args.save_path, "dict_val_id"), 'w')
    f2.write(info_json2)
    new_val_id = {v: k for k, v in dict_val_id.items()}


    num_classes = len(dict_id_all)
    # 加载backbone,默认resnet50
    model = get_model(args.backbone, pretrained=False)
    if args.model_path != "":
        model.load_state_dict(torch.load(args.model_path, map_location=device), False)
    model.to(device)

    # dataset
    train_dataset = ArcDataset(train_csv_train, dict_id_all, args.data_train_path, args.w,
                               args.h, IsNew=False)
    val_dataset = ArcDataset(train_csv_val, dict_id_all, args.data_train_path, args.w,
                             args.h, IsNew=False, dict_train_id=dict_val_id)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # 开始验证，获取特征矩阵
    if args.Feature_train_path is None:
        Feature_train, target_train = get_feature(model, train_loader, device, 512, get_id=False)
        Feature_train = Feature_train.cpu().detach().numpy()
        target_train = target_train.cpu().detach().numpy()
        # Img_id_train = Img_id_train.cpu().detach().numpy()
        np.save(os.path.join(args.save_path, "Feature_train.npy"), Feature_train)
        np.save(os.path.join(args.save_path, "target_train.npy"), target_train)
        # np.save(os.path.join(args.save_path, "Img_id_train.npy"), Img_id_train)
    else:
        Feature_train = np.load(args.Feature_train_path)
        target_train = np.load(args.target_train_path)
        # Img_id_train = np.load(args.Img_id_train_path)
    if args.Feature_val_path is None:
        Feature_val, target_val, Img_id_val = get_feature(model, val_loader, device, 512, get_id=True)
        target_val = target_val.cpu().detach().numpy()
        Img_id_val = Img_id_val.cpu().detach().numpy()
        np.save(os.path.join(args.save_path, "Feature_val.npy"), Feature_val.cpu().detach().numpy())
        np.save(os.path.join(args.save_path, "target_val.npy"), target_val)
        np.save(os.path.join(args.save_path, "Img_id_val.npy"), Img_id_val)
    else:
        Feature_val = torch.from_numpy(np.load(args.Feature_val_path))
        target_val = np.load(args.target_val_path)
        Img_id_val = np.load(args.Img_id_val_path)
    # 计算验证得分

    analyse = make_val(Feature_train, target_train, Feature_val, target_val, device, num_classes, Img_id_val,
                       new_id_all, new_val_id, train_csv_val)
    analyse.to_csv(os.path.join(args.save_path, "analyse.csv"), index=False)
    print(analyse.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练参数设置')
    parser.add_argument('--backbone', type=str, default='resnet50', help='特征网络选择，默认resnet50')
    parser.add_argument('--data_train_path', type=str, help='训练集路径', required=True)
    parser.add_argument('--save_path', type=str, help='存储路径', default=r'./')
    parser.add_argument('--dict_id_path', type=str, help='训练类型对应字典路径', required=True)
    parser.add_argument('--train_csv_train_path', type=str, help='需要训练数据csv路径', required=True)
    parser.add_argument('--train_csv_val_path', type=str, help='需要测试数据csv路径', required=True)
    parser.add_argument('--model_path', type=str, help='上次训练模型权重', required=True)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, help='冻结训练batch size', default=256)
    parser.add_argument('--Feature_train_path', type=str, help='训练集特征矩阵路径', default=None)
    parser.add_argument('--target_train_path', type=str, help='训练集标签矩阵路径', default=None)
    parser.add_argument('--Feature_val_path', type=str, help='验证集特征矩阵路径', default=None)
    parser.add_argument('--target_val_path', type=str, help='验证集标签矩阵路径', default=None)
    parser.add_argument('--Img_id_val_path', type=str, help='验证集image_id矩阵路径', default=None)
    parser.add_argument('--w', type=int, help='训练图片宽度', default=224)
    parser.add_argument('--h', type=int, help='训练图片高度', default=224)
    args = parser.parse_args()

    analyse(args)
