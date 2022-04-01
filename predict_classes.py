from __init__ import *

if __name__ == '__main__':
    # -------------------
    # 参数设置
    # -------------------
    model_path = r"D:\edge\resnet50_epoch_50_loss_0.02979323887588954"
    data_test_path = r'D:\project\happyWhale\archive\archive'
    path_0 = r"C:\Users\12529\Desktop\0"
    path_1 = r"C:\Users\12529\Desktop\1"
    # path_2 = r"C:\Users\12529\Desktop\2"
    backbone = 'resnet50'
    w = 224
    h = 224
    batch_size = 256
    num_workers = 4
    # -------------------------------#
    #   训练设备、网络、loss函数、dataset
    # -------------------------------#

    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("backbone = " + backbone)

    if backbone == 'EfficientNet-V2':
        model = timm.create_model('efficientnetv2_rw_m', pretrained=False, num_classes=3)
    else:
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    if model_path != "":
        model.load_state_dict(torch.load(model_path, map_location=device), False)

    model.to(device)
    model.eval()

    path_list = os.listdir(data_test_path)
    # 建立test_dataloader的csv文件
    submission = pd.DataFrame(columns=['image', 'predictions'])
    for item in range(len(path_list)):
        submission.loc[item, "image"] = path_list[item]
    # 建立测试集地址字典
    dict_id_test = dict(zip(path_list, range(len(path_list))))
    new_d_test = {v: k for k, v in dict_id_test.items()}

    test_dataset = TestDataset(submission, dict_id_test, data_test_path, w, h)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)

    Feature_test, target_test = get_feature(model, test_dataloader, device, 2)
    # target_test = target_test.cpu().detach().numpy()
    # Feature_test =  Feature_test.cpu().detach().numpy()
    with tqdm(total=target_test.shape[0], postfix=dict) as pbar:
        for item in range(target_test.shape[0]):
            Feature = Feature_test[item, :]
            Str = path_list[item]
            if Feature.argmax() == 0:
                shutil.copyfile(os.path.join(data_test_path, Str),
                                os.path.join(path_0, Str))
            elif Feature.argmax() == 1:
                shutil.copyfile(os.path.join(data_test_path, Str),
                                os.path.join(path_1, Str))
            pbar.update(1)

