from __init__ import *


def go_train(backbone, data_train_path, save_path,
             dict_id_path, train_csv_train_path,
             model_path, metric, num_workers,
             save_interval, Freeze_Epoch, Freeze_lr,
             Freeze_lr_step, Freeze_batch_size,
             Unfreeze_Epoch, Unfreeze_lr, Unfreeze_lr_step,
             Unfreeze_batch_size, w, h, pretrained, Freeze_weight_decay,
             Unfreeze_weight_decay):
    # 训练设备
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 清洗数据，生成训练所需csv及dict
    # train_csv_train, train_csv_val, dict_id_all = make_csv(data_csv_path,
    #                                                        low,
    #                                                        high,
    #                                                        val_number,
    #                                                        save_path)
    f2 = open(dict_id_path, 'r')
    dict_id_all = json.load(f2)
    train_csv_train = pd.read_csv(train_csv_train_path)
    train_csv_val = None

    num_classes = len(dict_id_all)

    # 加载模型的loss函数类型
    criterion = FocalLoss(gamma=2)

    # 加载backbone,默认resnet50
    if backbone == 'EfficientNet-V2':
        model = timm.create_model('efficientnetv2_rw_m', pretrained=pretrained, num_classes=512)
    else:
        model = torchvision.models.resnet50(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, 512)
    if model_path != "":
        model.load_state_dict(torch.load(model_path, map_location=device), False)

    # 加载模型的margin类型
    if metric == 'Arc':
        metric_fc = ArcMarginProduct(512, num_classes)
    elif metric == 'Add':
        metric_fc = AddMarginProduct(512, num_classes)
    else:
        metric_fc = SphereProduct(512, num_classes)

    # dataset
    train_dataset = ArcDataset(train_csv_train, dict_id_all, data_train_path, w,
                               h)
    if train_csv_val is None:
        val_dataset = ArcDataset(train_csv_val, dict_id_all, data_train_path, w,
                                 h)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=Freeze_batch_size, shuffle=True,
                                    num_workers=num_workers)
    else:
        val_dataloader = None

    # 训练前准备
    model.to(device)

    metric_fc.to(device)

    criterion.to(device)

    if Freeze_Epoch != 0:
        # -------------------------------#
        #   开始冻结训练
        # -------------------------------#
        print("--------冻结训练--------")
        # -------------------------------#
        #   选择优化器
        # -------------------------------#
        Freeze_optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                           lr=Freeze_lr, weight_decay=Freeze_weight_decay)
        Freeze_scheduler = StepLR(Freeze_optimizer, step_size=Freeze_lr_step, gamma=0.2)
        # -------------------------------#
        #   生成冻结dataloader
        # -------------------------------#
        Freeze_train_dataloader = DataLoader(dataset=train_dataset, batch_size=Freeze_batch_size, shuffle=True,
                                             num_workers=num_workers)
        # -------------------------------#
        #   冻结措施
        # -------------------------------#
        for param in model.parameters():
            param.requires_grad = False
        model = make_train(model=model,
                           metric_fc=metric_fc,
                           criterion=criterion,
                           optimizer=Freeze_optimizer,
                           scheduler=Freeze_scheduler,
                           train_loader=Freeze_train_dataloader,
                           val_loader=val_dataloader,
                           device=device,
                           num_classes=num_classes,
                           max_epoch=Freeze_Epoch + Unfreeze_Epoch,
                           save_interval=save_interval,
                           save_path=save_path,
                           backbone=backbone,
                           epoch_start=1,
                           epoch_end=Freeze_Epoch,
                           Str=metric,
                           Freeze_Epoch=Freeze_Epoch)

    # -------------------------------#
    #   开始解冻训练
    # -------------------------------#
    print("--------解冻训练--------")
    # -------------------------------#
    #   选择优化器
    # -------------------------------#
    Unfreeze_optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                         lr=Unfreeze_lr, weight_decay=Unfreeze_weight_decay)
    Unfreeze_scheduler = StepLR(Unfreeze_optimizer, step_size=Unfreeze_lr_step, gamma=0.2)
    # -------------------------------#
    #   生成解冻dataloader
    # -------------------------------#
    Unfreeze_train_dataloader = DataLoader(dataset=train_dataset, batch_size=Unfreeze_batch_size, shuffle=True,
                                           num_workers=num_workers)
    # -------------------------------#
    #   解冻措施
    # -------------------------------#
    for param in model.parameters():
        param.requires_grad = True
    make_train(model=model,
               metric_fc=metric_fc,
               criterion=criterion,
               optimizer=Unfreeze_optimizer,
               scheduler=Unfreeze_scheduler,
               train_loader=Unfreeze_train_dataloader,
               val_loader=val_dataloader,
               device=device,
               num_classes=num_classes,
               max_epoch=Freeze_Epoch + Unfreeze_Epoch,
               save_interval=save_interval,
               save_path=save_path,
               backbone=backbone,
               epoch_start=Freeze_Epoch + 1,
               epoch_end=Freeze_Epoch + Unfreeze_Epoch,
               Str=metric,
               Freeze_Epoch=Freeze_Epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练参数设置')
    parser.add_argument('--backbone', type=str, default='resnet50', help='特征网络选择，默认resnet50')
    parser.add_argument('--data_train_path', type=str, help='训练集路径', required=True)
    parser.add_argument('--data_csv_path', type=str, help='全体训练集csv路径', default=r'../input/happy-whale-and-dolphin/train.csv')
    parser.add_argument('--save_path', type=str, help='存储路径', default=r'./')
    parser.add_argument('--dict_id_path', type=str, help='训练类型对应字典路径', required=True)
    parser.add_argument('--train_csv_train_path', type=str, help='需要训练数据csv路径', required=True)
    parser.add_argument('--metric', type=str, help='Arc/Add/Sph', default='Arc')
    parser.add_argument('--pretrained', type=bool, help='是否需要预训练', default=True)
    parser.add_argument('--model_path', type=str, help='上次训练模型权重', default=r'')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_interval', type=int, help='保存间隔', default=3)
    parser.add_argument('--Freeze_Epoch', type=int, help='冻结训练轮次', default=12)
    parser.add_argument('--Freeze_lr', type=float, help='冻结训练lr', default=0.1)
    parser.add_argument('--Freeze_lr_step', type=int, help='冻结训练lr衰减周期', default=10)
    parser.add_argument('--Freeze_weight_decay', type=float, help='冻结训练权重衰减率', default=5e-4)
    parser.add_argument('--Freeze_batch_size', type=int, help='冻结训练batch size', default=256)
    parser.add_argument('--Unfreeze_Epoch', type=int, help='解冻训练轮次', default=36)
    parser.add_argument('--Unfreeze_lr', type=float, help='解冻训练lr', default=0.05)
    parser.add_argument('--Unfreeze_lr_step', type=int, help='解冻训练lr衰减周期', default=10)
    parser.add_argument('--Unfreeze_weight_decay', type=float, help='解冻训练权重衰减率', default=5e-4)
    parser.add_argument('--Unfreeze_batch_size', type=int, help='解冻训练batch size', default=64)
    parser.add_argument('--w', type=int, help='训练图片宽度', default=224)
    parser.add_argument('--h', type=int, help='训练图片高度', default=224)
    args = parser.parse_args()

    go_train(backbone=args.backbone,
             data_train_path=args.data_train_path,
             save_path=args.save_path,
             dict_id_path=args.dict_id_path,
             train_csv_train_path=args.train_csv_train_path,
             metric=args.metric,
             pretrained=args.pretrained,
             num_workers=args.num_workers,
             save_interval=args.save_interval,
             Freeze_Epoch=args.Freeze_Epoch,
             Freeze_lr=args.Freeze_lr,
             Freeze_lr_step=args.Freeze_lr_step,
             Freeze_weight_decay=args.Freeze_weight_decay,
             Freeze_batch_size=args.Freeze_batch_size,
             Unfreeze_Epoch=args.Unfreeze_Epoch,
             Unfreeze_lr=args.Unfreeze_lr,
             model_path=args.model_path,
             Unfreeze_lr_step=args.Unfreeze_lr_step,
             Unfreeze_weight_decay=args.Unfreeze_weight_decay,
             Unfreeze_batch_size=args.Unfreeze_batch_size,
             w=args.w,
             h=args.h)
    # # -------------------------------#
    # #   参数设置
    # # -------------------------------#
    # backbone = 'resnet'
    # # -------------------------------#
    # #   数据路径
    # # -------------------------------#
    # data_train_path = r'../input/happywhale-nohead-data/no_head/no_head'
    # data_csv_path = r'../input/happy-whale-and-dolphin/train.csv'
    # save_path = r'./'
    # dict_id_path = r'../input/happywhale-nohead-data/dict_id'
    # train_csv_train_path = r'../input/happywhale-nohead-data/train_csv_train.csv'
    # # -------------------------------#
    # #   model及设置
    # # -------------------------------#
    # model_path = r''
    # metric = 'Arc'
    # pretrained = True
    # num_workers = 2
    # save_interval = 2
    # # -------------------------------#
    # #   冻结训练
    # # -------------------------------#
    # Freeze_Epoch = 20
    # Freeze_lr = 1e-1
    # Freeze_lr_step = 10
    # Freeze_lr_decay = 0.95  # when val_loss increase lr = lr*lr_decay
    # Freeze_weight_decay = 5e-4
    # Freeze_batch_size = 128
    # # -------------------------------#
    # #   解冻训练
    # # -------------------------------#
    # Unfreeze_Epoch = 50
    # Unfreeze_lr = 0.1  # initial learning rate
    # Unfreeze_lr_step = 10
    # Unfreeze_lr_decay = 0.95  # when val_loss increase lr = lr*lr_decay
    # Unfreeze_weight_decay = 5e-4
    # Unfreeze_batch_size = 64
    # # -------------------------------#
    # #   分类数量，及输入图像设计
    # # -------------------------------#
    # w = 512
    # h = 512
    # low = 0
    # high = 1000
    # val_number = 0
