from __init__ import *


if __name__ == '__main__':
    # -------------------------------#
    #   路径设置
    # -------------------------------#
    data_root_path = '../input/humpback-whale-identification'
    save_path = r'./'
    path = r'../input/arc-epoth-2'
    # -------------------------------#
    #   dataloader设置
    # -------------------------------#
    w = 256
    h = 256
    num_workers = 2
    batch_size = 128
    # -------------------------------#
    #   开始预测
    # -------------------------------#
    with torch.no_grad():
        data_test_path = os.path.join(data_root_path, "test")
        data_csv_path = os.path.join(data_root_path, "train.csv")
        print(torch.cuda.is_available())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 建立全局字典
        train_csv = pd.read_csv(data_csv_path)
        train_csv_id = train_csv['Id'].unique()
        dict_id_all = dict(zip(train_csv_id, range(len(train_csv_id))))
        new_d_all = {v: k for k, v in dict_id_all.items()}

        model, dict_id, Feature_train, target_train = get_pre_need(path, device)
        model.eval()
        # 获取各个总类中心点
        Feature_train_num = np.zeros([len(dict_id), 512])
        for item in range(len(dict_id)):
            Feature_train_num[item] = np.mean(Feature_train[target_train[:, 0] == item, :], axis=0)

        path_list = os.listdir(os.path.join(data_root_path, "test"))
        # 建立test_dataloader的csv文件
        submission = pd.DataFrame(columns=['Image', 'Id'])
        for item in range(len(path_list)):
            submission.loc[item, "Image"] = path_list[item]
        # 建立测试集地址字典
        dict_id_test = dict(zip(path_list, range(len(path_list))))
        new_d_test = {v: k for k, v in dict_id_test.items()}

        test_dataset = TestDataset(submission, dict_id_test, data_test_path, w, h)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers)

        Feature_test, target_test = get_feature(model, test_dataloader, device)

        target_test = target_test.cpu().detach().numpy()
        with tqdm(total=target_test.shape[0], postfix=dict) as pbar:
            for item in range(target_test.shape[0]):
                # Top, Top_index = get_pre(Feature_test[item, :], Feature_train, target_train, dict_id,
                #                          dict_id_all,
                #                          4, device)
                Top, Top_index = get_pre_num(Feature_test[item, :], Feature_train_num, dict_id, dict_id_all, 5, device)
                # Is_new = 'new_whale' if Top[0] < 0.75 else new_d_all[Top_index[4]]
                submission.loc[submission[submission.Image == new_d_test[target_test[item, 0]]].index.tolist(), "Id"] = \
                    new_d_all[Top_index[0]] + ' ' + 'new_whale'+ ' ' + new_d_all[Top_index[1]] + ' ' + new_d_all[Top_index[
                        2]] + ' ' + new_d_all[Top_index[3]]
                # Top[4] = 0.55
                # Top_index[4] = dict_id_all['new_whale']
                # Top_index = Top_index[np.argsort(-Top)]
                # # Is_new = 'new_whale' if Top[0] < 0.75 else new_d_all[Top_index[4]]
                # submission.loc[submission[submission.Image == new_d_test[target_test[item, 0]]].index.tolist(), "Id"] = \
                #     new_d_all[Top_index[0]] + ' ' + new_d_all[Top_index[1]] + ' ' + new_d_all[Top_index[
                #         2]] + ' ' + new_d_all[Top_index[3]] + ' ' + new_d_all[Top_index[4]]
                pbar.update(1)
                pbar.set_postfix(
                    **{'Top': Top[0]})
        submission.to_csv(os.path.join(save_path, "submission.csv"), index=False)
