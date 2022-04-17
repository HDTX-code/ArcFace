from __init__ import *

if __name__ == '__main__':
    # -------------------
    # 参数设置
    # -------------------
    model_path = r"D:\edge\resnet18Softmax-12loss_ 0.10518473038102259score_ 0.9573673870333989.pth"
    dict_id_path = r"D:\project\happyWhale\weights\res50-softmax-224\dict_id"
    data_test_path = r"D:\project\happyWhale\classes\CFL\test\test_all\test_all"
    data_path_list = [r"C:\Users\12529\Desktop\beluga", r"C:\Users\12529\Desktop\big_model",
                      r"C:\Users\12529\Desktop\hdb"]
    # path_0 = r"C:\Users\12529\Desktop\0"
    # path_1 = r"C:\Users\12529\Desktop\1"
    # path_2 = r"C:\Users\12529\Desktop\2"
    backbone = 'resnet18'
    w = 384
    h = 384
    batch_size = 256
    num_workers = 4
    num_classes = len(data_path_list)
    # -------------------------------#
    #   训练设备、网络、loss函数、dataset
    # -------------------------------#

    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("backbone = " + backbone)

    model = get_model(backbone, False, 30)
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
    new_d_test = {v: k for k, v in dict_id_test.items()}

    test_dataset = TestDataset(submission, dict_id_test, data_test_path, w, h)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)

    Feature_test, target_test = get_feature(model, test_dataloader, device, num_classes)
    target_test = target_test.cpu().detach().numpy()
    Feature_test = Feature_test.cpu().detach().numpy()

    for item in data_path_list:
        if not os.path.exists(item):
            os.mkdir(item)

    with tqdm(total=target_test.shape[0], postfix=dict) as pbar:
        for item in range(target_test.shape[0]):
            Feature = Feature_test[item, :]
            Str = new_d_test[target_test[item, 0]]
            if np.argmax(Feature) == 1 or np.argmax(Feature) == 7 or np.argmax(Feature) == 3:
                shutil.copyfile(os.path.join(data_test_path, Str),
                                os.path.join(data_path_list[2], Str))
            elif np.argmax(Feature) == 4:
                shutil.copyfile(os.path.join(data_test_path, Str),
                                os.path.join(data_path_list[0], Str))
            else:
                shutil.copyfile(os.path.join(data_test_path, Str),
                                os.path.join(data_path_list[1], Str))
            pbar.update(1)
