from __init__ import *


def go_predict(a, data_root_path, save_path, path_0_2, path_3_11, path_12_1000):
    with torch.no_grad():
        opt = Config()
        opt.data_test_path = os.path.join(data_root_path, "test")
        print(torch.cuda.is_available())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_0_2, dict_id_0_2, Feature_train_0_2, target_train_0_2 = get_pre_need(path_0_2, device)
        model_0_2.eval()
        model_3_11, dict_id_3_11, Feature_train_3_11, target_train_3_11 = get_pre_need(path_3_11, device)
        model_3_11.eval()
        model_12_1000, dict_id_12_1000, Feature_train_12_1000, target_train_12_1000 = get_pre_need(path_12_1000, device)
        model_12_1000.eval()

        path_list = os.listdir(os.path.join(data_root_path, "test"))
        # 建立test_dataloader的csv文件
        submission = pd.DataFrame(columns=['Image', 'Id'])
        for item in range(len(path_list)):
            submission.loc[item, "Image"] = path_list[item]
        # 建立全局字典
        dict_id_all = dict(zip(path_list, range(len(path_list))))
        new_d_all = {v: k for k, v in dict_id_all.items()}

        test_dataset = TestDataset(opt, submission, dict_id_all)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=True,
                                     num_workers=opt.num_workers)

        Feature_test_0_2, target_test_0_2 = get_feature(model_0_2, test_dataloader, device)
        Feature_test_3_11, target_test_3_11 = get_feature(model_3_11, test_dataloader, device)
        Feature_test_12_1000, target_test_12_1000 = get_feature(model_12_1000, test_dataloader, device)

        with tqdm(total=target_test_0_2.shape[0], postfix=dict) as pbar:
            for item in range(target_test_0_2.shape[0]):

                Top_0_2, Top_index_0_2 = get_pre(Feature_test_0_2[item, :], Feature_train_0_2, target_train_0_2, dict_id_0_2, dict_id_all,
                                                 4, device)
                Top_3_11, Top_index_3_11 = get_pre(Feature_test_3_11[item, :], Feature_train_3_11, target_train_3_11, dict_id_3_11, dict_id_all
                                                   , 4, device)
                Top_12_1000, Top_index_12_1000 = get_pre(Feature_test_12_1000[item, :], Feature_train_12_1000,
                                                         target_train_12_1000, dict_id_12_1000, dict_id_all,
                                                         4, device)
                Top = np.concatenate((Top_0_2, Top_3_11, Top_12_1000), axis=0)
                Top_index = np.concatenate((Top_index_0_2, Top_index_3_11, Top_index_12_1000), axis=0)
                Top_index = Top_index[np.argsort(-Top)[0:4]]
                Top = Top[Top_index]
                submission.loc[item, "Id"] = new_d_all[Top_index[0]] + ' ' + new_d_all[Top_index[1]] + ' ' + new_d_all[Top_index[
                    2]] + ' ' + new_d_all[Top_index[3]] + ' ' + 'new_whale'
                pbar.update(1)
                pbar.set_postfix(
                    **{'Top': Top})
        submission.to_csv(os.path.join(save_path, "submission.csv"), index=False)


if __name__ == '__main__':
    go_predict(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    # go_predict(0, r"D:\project\humpWhale\data\humpback-whale-identification", r"D:\project\humpWhale\data",
    #            r"D:\project\humpWhale\data\0-2", r"D:\project\humpWhale\data\3-11", r"D:\project\humpWhale\data\12-1000")
