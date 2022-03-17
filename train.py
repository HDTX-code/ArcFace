from __init__ import *


def go_train(a, dict_id_path, data_root_path, save_root_path, low, high, val_number):
    # 生成公共参数类
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
    train_csv_train, train_csv_val, num = make_csv(opt)
    dict_id = np.load(dict_id_path).item()
    new_d = {v: k for k, v in dict_id.items()}
    opt.num_classes = len(num)

    # 生成train、val的dataloader
    train_dataset = ArcDataset(opt, train_csv_train, dict_id)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.num_workers)
    val_dataset = ArcDataset(opt, train_csv_val, dict_id)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=True,
                                num_workers=opt.num_workers)

    # 加载模型的loss函数类型
    criterion = FocalLoss(gamma=2)

    # 加载backbone
    model_Sph = torchvision.models.resnet50(pretrained=True)
    model_Sph.fc = torch.nn.Linear(model_Sph.fc.in_features, 512)
    # model_Sph.load_state_dict( torch.load("D:\\project\\humpWhale\\arcFace\\ArcFace-modification-\\log\\gt30_model
    # .pth", map_location=device), False)

    # 加载模型的margin类型
    metric_fc_Sph = SphereProduct(512, opt.num_classes, m=4)

    # 选择优化器
    optimizer_Sph = torch.optim.SGD([{'params': model_Sph.parameters()}, {'params': metric_fc_Sph.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)

    # 训练前准备
    model_Sph.to(device)

    metric_fc_Sph.to(device)

    criterion.to(device)

    scheduler_Sph = StepLR(optimizer_Sph, step_size=opt.lr_step, gamma=0.1)

    # 开始训练
    make_train(model_Sph, metric_fc_Sph, criterion, optimizer_Sph, scheduler_Sph, train_dataloader,
               val_dataloader, opt, device, num, "Sph")


if __name__ == '__main__':
    go_train(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
