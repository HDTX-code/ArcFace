from __init__ import *
from models.metrics import AddMarginProduct


def go_train(a, data_root_path, save_root_path, low, high, val_number, max_epoch, IsTrain, isArc):
    # 生成公共参数类
    opt = Config()
    opt.data_train_path = os.path.join(data_root_path, "train")
    opt.data_csv_path = os.path.join(data_root_path, "train.csv")
    opt.data_test_path = os.path.join(data_root_path, "test")
    opt.checkpoints_path = save_root_path
    opt.low = int(low)
    opt.high = int(high)
    opt.val_number = int(val_number)
    opt.max_epoch = int(max_epoch)
    # 训练设备
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 清洗数据，生成训练所需csv及dict
    train_csv_train, train_csv_val, dict_id_all = make_csv(opt, save_root_path)
    opt.num_classes = len(dict_id_all)

    # 生成train、val的dataloader
    train_dataset = ArcDataset(opt, train_csv_train, dict_id_all)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.num_workers)
    if val_number != 0:
        val_dataset = ArcDataset(opt, train_csv_val, dict_id_all)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=True,
                                    num_workers=opt.num_workers)
    else:
        val_dataloader = None

    # 加载模型的loss函数类型
    criterion = FocalLoss(gamma=2)

    # 加载backbone
    if IsTrain == "":
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 512)
    else:
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 512)
        model.load_state_dict(torch.load(IsTrain, map_location=device), False)

        # 加载模型的margin类型
    if int(isArc) == 0:
        metric_fc = ArcMarginProduct(512, opt.num_classes)
        str = 'Arc'
    elif int(isArc) == 1:
        metric_fc = AddMarginProduct(512, opt.num_classes)
        str = 'Add'
    else:
        metric_fc = SphereProduct(512, opt.num_classes)
        str = 'Sph'

    # 选择优化器
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                lr=opt.lr, weight_decay=opt.weight_decay)

    # 训练前准备
    model.to(device)

    metric_fc.to(device)

    criterion.to(device)

    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    # 开始训练
    make_train(model, metric_fc, criterion, optimizer, scheduler, train_dataloader,
               val_dataloader, opt, device, len(dict_id_all), str)


if __name__ == '__main__':
    go_train(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
