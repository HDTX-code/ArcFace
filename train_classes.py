import torchvision.models

from __init__ import *


def go_train_classes(backbone, path_list, save_path, model_path, num_workers,
                     save_interval, Freeze_Epoch, Freeze_lr,
                     Freeze_lr_step, Freeze_batch_size,
                     Unfreeze_Epoch, Unfreeze_lr, Unfreeze_lr_step,
                     Unfreeze_batch_size, w, h, pretrained, Freeze_weight_decay,
                     Unfreeze_weight_decay):
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("backbone = " + backbone)

    num_classes = len(path_list)
    print("num_classes = " + num_classes)

    if backbone == 'EfficientNet-V2':
        model = timm.create_model('efficientnetv2_rw_m', pretrained=pretrained, num_classes=num_classes)
    elif backbone == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        model = torchvision.models.resnet34(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if model_path != "":
        model.load_state_dict(torch.load(model_path, map_location=device), False)

    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # 获取csv
    train_csv = pd.DataFrame(columns=['image', 'path', 'individual_id'])
    for item in range(len(path_list)):
        csv = get_csv(path_list[item], item)
        train_csv = pd.concat([train_csv, csv], ignore_index=True)

    # 生成dataset
    train_dataset = ClassesDataset(train_csv, w, h)

    if Freeze_Epoch != 0:
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
    for item in range(Freeze_Epoch + 1, Freeze_Epoch + Unfreeze_Epoch + 1):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练训练参数设置')
    parser.add_argument('--backbone', type=str, default='resnet18', help='特征网络选择，默认resnet18')
    parser.add_argument('--path_list', type=str, nargs='+', help='分类数据集路径', required=True)
    parser.add_argument('--save_path', type=str, help='存储路径', default=r'./')
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

    go_train_classes(backbone=args.backbone,
                     save_path=args.save_path,
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
                     path_list=args.path_list,
                     w=args.w,
                     h=args.h)
    # # -------------------
    # # 路径设置
    # # -------------------
    # # path_0 = r"../input/happywhale-classes-4/0"
    # # path_1 = r"../input/happywhale-classes-4/1"
    # # path_2 = r"../input/happywhale-classes/2"
    # save_path = r''
    # model_path = r'../input/classes-weights-3/resnet18_epoch_36_loss_0.05083953045509957'
    # # -------------------
    # # 网络设置
    # # -------------------
    # backbone = 'resnet18'
    # pretrained = True
    # num_workers = 2
    # save_interval = 2
    # # -------------------------------#
    # #   冻结训练
    # # -------------------------------#
    # Freeze_Epoch = 0
    # Freeze_lr = 1e-1
    # Freeze_lr_step = 10
    # Freeze_lr_decay = 0.95  # when val_loss increase lr = lr*lr_decay
    # Freeze_weight_decay = 5e-4
    # Freeze_batch_size = 2
    # # -------------------------------#
    # #   解冻训练
    # # -------------------------------#
    # Unfreeze_Epoch = 16
    # Unfreeze_lr = 1e-2  # initial learning rate
    # Unfreeze_lr_step = 4
    # Unfreeze_lr_decay = 0.95  # when val_loss increase lr = lr*lr_decay
    # Unfreeze_weight_decay = 5e-4
    # Unfreeze_batch_size = 128
    # # -------------------------------#
    # #   分类数量，及输入图像设计
    # # -------------------------------#
    # w = 512
    # h = 512
    # num_classes = 2
    # # -------------------------------#
    # #   训练设备、网络、loss函数、dataset
    # # -------------------------------#
