class Config(object):
    # 特征提取网络
    backbone = 'resnet50'
    # 数据路径
    data_train_path = 'D:\\project\\humpWhale\\arcFace\\ArcFace-modification-\\data\\train'
    data_csv_path = 'D:\\project\\humpWhale\\arcFace\\ArcFace-modification-\\data\\train.csv'
    data_test_path = 'D:\\project\\humpWhale\\arcFace\\ArcFace-modification-\\data\\test'
    checkpoints_path = 'D:\\project\\humpWhale\\arcFace\\ArcFace-modification-\\log'
    # 分类数量
    num_classes = 2
    # 转换后图片大小
    w = 256
    h = 256
    # Loss函数类型
    loss = 'focal_loss'
    # margin类型
    metric = 'add_margin'
    #
    use_se = False
    #
    easy_margin = False
    # 是否加载预训练参数
    pretrained = True
    # 优化器类型
    optimizer = 'sgd'
    # 训练参数
    max_epoch = 21
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    batch_size = 64
    num_workers = 2
    # 模型保存参数
    save_interval = 3
    # Canny阈值
    th1 = 300
    th2 = 600
    # 选取数据量范围的上下限
    low = 15
    high = 30
    # 验证集每类的数量
    val_number = 2
