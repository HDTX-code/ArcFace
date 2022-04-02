from __init__ import *

if __name__ == '__main__':
    # -------------------
    # 路径设置
    # -------------------
    path_0 = r"../input/happywhale-classes-4/0"
    path_1 = r"../input/happywhale-classes-4/1"
    # path_2 = r"../input/happywhale-classes/2"
    save_path = r''
    model_path = r'../input/classes-weights-3/resnet18_epoch_36_loss_0.05083953045509957'
    # -------------------
    # 网络设置
    # -------------------
    backbone = 'resnet18'
    pretrained = True
    num_workers = 2
    save_interval = 2
    # -------------------------------#
    #   冻结训练
    # -------------------------------#
    Freeze = False
    Freeze_Epoch = 20
    Freeze_lr = 1e-1
    Freeze_lr_step = 10
    Freeze_lr_decay = 0.95  # when val_loss increase lr = lr*lr_decay
    Freeze_weight_decay = 5e-4
    Freeze_batch_size = 2
    # -------------------------------#
    #   解冻训练
    # -------------------------------#
    Unfreeze_Epoch = 16
    Unfreeze_lr = 1e-2  # initial learning rate
    Unfreeze_lr_step = 4
    Unfreeze_lr_decay = 0.95  # when val_loss increase lr = lr*lr_decay
    Unfreeze_weight_decay = 5e-4
    Unfreeze_batch_size = 128
    # -------------------------------#
    #   分类数量，及输入图像设计
    # -------------------------------#
    w = 512
    h = 512
    num_classes = 2
    # -------------------------------#
    #   训练设备、网络、loss函数、dataset
    # -------------------------------#

    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("backbone = " + backbone)

    if backbone == 'EfficientNet-V2':
        model = timm.create_model('efficientnetv2_rw_m', pretrained=pretrained, num_classes=num_classes)
    elif backbone == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        model = torchvision.models.alexnet(pretrained=pretrained)
        num_ftrs = model.classifier[0].in_features
        model.classifier[0] = nn.Linear(num_ftrs, num_classes)
    if model_path != "":
        model.load_state_dict(torch.load(model_path, map_location=device), False)

    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # 获取csv
    csv_0 = get_csv(path_0, 0)
    csv_1 = get_csv(path_1, 1)
    # csv_2 = get_csv(path_2, 2)
    train_csv = pd.concat([csv_0, csv_1], ignore_index=True)

    # 生成dataset
    train_dataset = ClassesDataset(train_csv, w, h)

    if Freeze:
        # -------------------------------#
        #   开始冻结训练
        # -------------------------------#
        print("--------冻结训练--------")
        # -------------------------------#
        #   选择优化器
        # -------------------------------#
        Freeze_optimizer = torch.optim.SGD(model.parameters(), lr=Freeze_lr, weight_decay=Freeze_weight_decay)
        Freeze_scheduler = StepLR(Freeze_optimizer, step_size=Freeze_lr_step, gamma=0.1)
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
        for item in range(1, Freeze_Epoch + 1):
            fit_one_epoch_classes(model=model,
                                  criterion=criterion,
                                  optimizer=Freeze_optimizer,
                                  item=item,
                                  max_epoch=Unfreeze_Epoch + Freeze_Epoch,
                                  Freeze_Epoch=Freeze_Epoch,
                                  train_loader=Freeze_train_dataloader,
                                  device=device,
                                  save_interval=save_interval,
                                  save_path=save_path,
                                  backbone=backbone,
                                  num_classes=num_classes)
            Freeze_scheduler.step()
        # -------------------------------#
        #   开始解冻训练
        # -------------------------------#
    print("--------解冻训练--------")
    # -------------------------------#
    #   选择优化器
    # -------------------------------#
    Unfreeze_optimizer = torch.optim.SGD(model.parameters(), lr=Unfreeze_lr, weight_decay=Unfreeze_weight_decay)
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
    for item in range(Freeze_Epoch+1, Freeze_Epoch + Unfreeze_Epoch + 1):
        fit_one_epoch_classes(model=model,
                              criterion=criterion,
                              optimizer=Unfreeze_optimizer,
                              item=item,
                              max_epoch=Unfreeze_Epoch + Freeze_Epoch,
                              Freeze_Epoch=Freeze_Epoch,
                              train_loader=Unfreeze_train_dataloader,
                              device=device,
                              save_interval=save_interval,
                              save_path=save_path,
                              backbone=backbone,
                              num_classes=num_classes)
        Unfreeze_scheduler.step()
