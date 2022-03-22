from __init__ import *


def go_train(a, data_root_path, save_root_path, low, high, val_number, max_epoch):
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
    # val_dataset = ArcDataset(opt, train_csv_val, dict_id_all)
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=True,
    #                             num_workers=opt.num_workers)
    val_dataloader = None

    # 加载模型的loss函数类型
    criterion = FocalLoss(gamma=2)

    # 加载backbone
    model_Arc = torchvision.models.resnet50(pretrained=True)
    model_Arc.fc = torch.nn.Linear(model_Arc.fc.in_features, 512)
    # model_Sph.load_state_dict( torch.load("D:\\project\\humpWhale\\arcFace\\ArcFace-modification-\\log\\gt30_model
    # .pth", map_location=device), False)

    # 加载模型的margin类型
    metric_fc_Arc = ArcMarginProduct(512, opt.num_classes)

    # 选择优化器
    optimizer_Arc = torch.optim.SGD([{'params': model_Arc.parameters()}, {'params': metric_fc_Arc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)

    # 训练前准备
    model_Arc.to(device)

    metric_fc_Arc.to(device)

    criterion.to(device)

    scheduler_Arc = StepLR(optimizer_Arc, step_size=opt.lr_step, gamma=0.1)

    # 开始训练
    make_train(model_Arc, metric_fc_Arc, criterion, optimizer_Arc, scheduler_Arc, train_dataloader,
               val_dataloader, opt, device, len(dict_id_all), "Arc")


if __name__ == '__main__':
    go_train(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
