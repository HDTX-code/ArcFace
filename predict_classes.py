from __init__ import *

if __name__ == '__main__':
    # -------------------
    # 参数设置
    # -------------------
    model_path = r"D:\project\happyWhale\classes\粗分类\weights\resnet18_epoch_30_loss_0.033053112030029294"
    data_test_path = r"D:\edge\archive"
    data_path_list = [r"C:\Users\12529\Desktop\0", r"C:\Users\12529\Desktop\1"]
    # path_0 = r"C:\Users\12529\Desktop\0"
    # path_1 = r"C:\Users\12529\Desktop\1"
    # path_2 = r"C:\Users\12529\Desktop\2"
    backbone = 'resnet18'
    w = 224
    h = 224
    batch_size = 256
    num_workers = 4
    num_classes = len(data_path_list)
    # -------------------------------#
    #   训练设备、网络、loss函数、dataset
    # -------------------------------#

    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("backbone = " + backbone)

    if backbone == 'EfficientNet-V2':
        model = timm.create_model('efficientnetv2_rw_m', pretrained=False, num_classes=3)
    else:
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if model_path != "":
        model.load_state_dict(torch.load(model_path, map_location=device), False)

    model.to(device)
    model.eval()

    path_list = os.listdir(data_test_path)
    # 建立test_dataloader的csv文件
    submission = pd.DataFrame(columns=['imag'])
    for item in range(len(path_list)):
        submission.loc[item, "image"] = path_list[item]
    # 建立测试集地址字典
    dict_id_test = dict(zip(path_list, range(len(path_list))))

    test_dataset = TestDataset(submission, dict_id_test, data_test_path, w, h)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)

    Feature_test, target_test = get_feature(model, test_dataloader, device, num_classes)

    for item in data_path_list:
        if not os.path.exists(item):
            os.mkdir(item)

    with tqdm(total=target_test.shape[0], postfix=dict) as pbar:
        for item in range(target_test.shape[0]):
            Feature = Feature_test[item, :]
            Str = path_list[item]
            shutil.copyfile(os.path.join(data_test_path, Str),
                            os.path.join(data_path_list[Feature.argmax()], Str))
            pbar.update(1)
