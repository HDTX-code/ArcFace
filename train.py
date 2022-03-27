import pandas as pd
import torch.optim

from __init__ import *

if __name__ == '__main__':
    # -------------------------------#
    #   参数设置
    # -------------------------------#
    backbone = 'EfficientNet-V2'
    # -------------------------------#
    #   数据路径
    # -------------------------------#
    data_train_path = r'../input/notebook6b97c908c9'
    data_csv_path = r'../input/happy-whale-and-dolphin/train.csv'
    save_path = r'./'
    dict_id_path = r'../input/raw-data/dict_id'
    train_csv_train_path = r'../input/raw-data/train_csv_train.csv'
    # -------------------------------#
    #   model及设置
    # -------------------------------#
    model_path = r''
    metric = 'Arc'
    pretrained = True
    num_workers = 6
    save_interval = 3
    # -------------------------------#
    #   冻结训练
    # -------------------------------#
    Freeze_Epoch = 6
    Freeze_lr = 4e-2
    Freeze_lr_step = 10
    Freeze_lr_decay = 0.95  # when val_loss increase lr = lr*lr_decay
    Freeze_weight_decay = 5e-4
    Freeze_batch_size = 256
    # -------------------------------#
    #   解冻训练
    # -------------------------------#
    Unfreeze_Epoch = 24
    Unfreeze_lr = 4e-3  # initial learning rate
    Unfreeze_lr_step = 10
    Unfreeze_lr_decay = 0.95  # when val_loss increase lr = lr*lr_decay
    Unfreeze_weight_decay = 5e-4
    Unfreeze_batch_size = 32
    # -------------------------------#
    #   分类数量，及输入图像设计
    # -------------------------------#
    w = 224
    h = 224
    low = 0
    high = 1000
    val_number = 0

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
    train_csv_train = train_csv_train[:10000]
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
    if int(val_number) != 0:
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

    # -------------------------------#
    #   开始冻结训练
    # -------------------------------#
    print("--------冻结训练--------")
    # -------------------------------#
    #   选择优化器
    # -------------------------------#
    Freeze_optimizer = torch.optim.RMSprop([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                           lr=Freeze_lr, weight_decay=Freeze_weight_decay)
    Freeze_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Freeze_optimizer, T_max=6, eta_min=4e-4)
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
    Unfreeze_optimizer = torch.optim.RMSprop([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                         lr=Unfreeze_lr, weight_decay=Unfreeze_weight_decay)
    Unfreeze_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Freeze_optimizer, T_max=12, eta_min=4e-4)
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
