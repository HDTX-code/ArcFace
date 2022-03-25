from __init__ import *
from models.metrics import AddMarginProduct


class Train(object):
    _defaults = {
        # 特征提取网络
        "backbone": 'resnet50',
        # 数据路径
        "data_train_path": '../input/humpback-whale-identification/train',
        "data_csv_path": '../input/humpback-whale-identification/train.csv',
        "data_test_path": "../input/humpback-whale-identification/test",
        "save_path": './',
        'model_path': "",
        # 分类数量
        "num_classes": 2,
        # 转换后图片大小
        "w": 256,
        "h": 256,
        # Loss函数类型
        "loss": 'focal_loss',
        # margin类型
        "metric": 'Arc',
        # 是否加载预训练参数
        "pretrained": True,
        # 优化器类型
        "optimizer": 'sgd',
        # 训练参数
        "Freeze_Epoch": 12,
        "Unfreeze_Epoch": 24,
        "lr": 1e-1,  # initial learning rate
        "lr_step": 10,
        "lr_decay": 0.95,  # when val_loss increase, lr : lr*lr_decay
        "weight_decay": 5e-4,
        "batch_size": 128,
        "num_workers": 2,
        # 模型保存参数
        "save_interval": 3,
        # 选取数据量范围的上下限
        "low": 0,
        "high": 1000,
        # 验证集每类的数量
        "val_number": 2,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

    def go_train(self):
        # 训练设备
        print(torch.cuda.is_available())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 清洗数据，生成训练所需csv及dict
        train_csv_train, train_csv_val, dict_id_all = make_csv(self._defaults["data_csv_path"],
                                                               self._defaults["low"],
                                                               self._defaults["high"],
                                                               self._defaults["val_number"],
                                                               self._defaults["save_path"])
        num_classes = len(dict_id_all)

        # 生成train、val的dataloader
        train_dataset = ArcDataset(train_csv_train, dict_id_all, self._defaults["data_train_path"], self._defaults["w"],
                                   self._defaults["h"])
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self._defaults["batch_size"], shuffle=True,
                                      num_workers=self._defaults["num_workers"])
        if int(self._defaults["val_number"]) != 0:
            val_dataset = ArcDataset(train_csv_val, dict_id_all, self._defaults["data_train_path"], self._defaults["w"],
                                     self._defaults["h"])
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=self._defaults["batch_size"], shuffle=True,
                                        num_workers=self._defaults["num_workers"])
        else:
            val_dataloader = None

        # 加载模型的loss函数类型
        criterion = FocalLoss(gamma=2)

        # 加载backbone,默认resnet50
        if self._defaults['backbone'] == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, 512)
        else:
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, 512)
        if self._defaults['model_path'] != "":
            model.load_state_dict(torch.load(self._defaults['model_path'], map_location=device), False)

            # 加载模型的margin类型
        if self._defaults["metric"] == 'Arc':
            metric_fc = ArcMarginProduct(512, num_classes)
        elif self._defaults["metric"] == 'Add':
            metric_fc = AddMarginProduct(512, num_classes)
        else:
            metric_fc = SphereProduct(512, num_classes)

        # 选择优化器
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=self._defaults["lr"], weight_decay=self._defaults['weight_decay'])

        # 训练前准备
        model.to(device)

        metric_fc.to(device)

        criterion.to(device)

        scheduler = StepLR(optimizer, step_size=self._defaults["lr_step"], gamma=0.1)

        # 开始训练,先冻结训练
        print("--------冻结训练--------")
        for param in model.parameters():
            param.requires_grad = False
        model = make_train(model=model,
                           metric_fc=metric_fc,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           train_loader=train_dataloader,
                           val_loader=val_dataloader,
                           device=device,
                           num_classes=num_classes,
                           max_epoch=self._defaults['Freeze_Epoch'] + self._defaults['Unfreeze_Epoch'],
                           save_interval=self._defaults['save_interval'],
                           save_path=self._defaults['save_path'],
                           backbone=self._defaults['backbone'],
                           epoch_start=1,
                           epoch_end=self._defaults['Freeze_Epoch'] + 1,
                           Str=self._defaults["metric"])
        print("--------解冻训练--------")
        for param in model.parameters():
            param.requires_grad = True
        make_train(model=model,
                   metric_fc=metric_fc,
                   criterion=criterion,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   train_loader=train_dataloader,
                   val_loader=val_dataloader,
                   device=device,
                   num_classes=num_classes,
                   max_epoch=self._defaults['Freeze_Epoch'] + self._defaults['Unfreeze_Epoch'],
                   save_interval=self._defaults['save_interval'],
                   save_path=self._defaults['save_path'],
                   backbone=self._defaults['backbone'],
                   epoch_start=self._defaults['Freeze_Epoch'] + 1,
                   epoch_end=self._defaults['Freeze_Epoch'] + self._defaults['Unfreeze_Epoch'] + 1,
                   Str=self._defaults["metric"])


def start_train(a, data_train_path, data_csv_path, save_path):
    train = Train(data_train_path=data_train_path,
                  data_csv_path=data_csv_path,
                  save_path=save_path)
    train.go_train()


if __name__ == '__main__':
    start_train(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3])
