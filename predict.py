from __init__ import *


def go_predict(a, data_root_path, save_path, path):
    with torch.no_grad():
        opt = Config()
        opt.data_test_path = os.path.join(data_root_path, "test")
        opt.data_csv_path = os.path.join(data_root_path, "train.csv")
        print(torch.cuda.is_available())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 建立全局字典
        train_csv = pd.read_csv(opt.data_csv_path)
        train_csv_id = train_csv['Id'].unique()
        dict_id_all = dict(zip(train_csv_id, range(len(train_csv_id))))
        new_d_all = {v: k for k, v in dict_id_all.items()}

        model, dict_id, Feature_train, target_train = get_pre_need(path, device)
        model.eval()

        path_list = os.listdir(os.path.join(data_root_path, "test"))
        # 建立test_dataloader的csv文件
        submission = pd.DataFrame(columns=['Image', 'Id'])
        for item in range(len(path_list)):
            submission.loc[item, "Image"] = path_list[item]
        # 建立测试集地址字典
        dict_id_test = dict(zip(path_list, range(len(path_list))))
        new_d_test = {v: k for k, v in dict_id_test.items()}

        test_dataset = TestDataset(opt, submission, dict_id_test)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False,
                                     num_workers=opt.num_workers)

        Feature_test, target_test = get_feature(model, test_dataloader, device)

        target_test = target_test.cpu().detach().numpy()
        with tqdm(total=target_test.shape[0], postfix=dict) as pbar:
            for item in range(target_test.shape[0]):
                Top, Top_index = get_pre(Feature_test[item, :], Feature_train, target_train, dict_id,
                                         dict_id_all,
                                         4, device)

                submission.loc[submission[submission.Image == new_d_test[target_test[item, 0]]].index.tolist(), "Id"] = \
                    new_d_all[Top_index[0]] + ' ' + new_d_all[Top_index[1]] + ' ' + new_d_all[Top_index[
                        2]] + ' ' + new_d_all[Top_index[3]] + ' ' + 'new_whale'
                pbar.update(1)
                pbar.set_postfix(
                    **{'Top': Top[3]})
        submission.to_csv(os.path.join(save_path, "submission.csv"), index=False)


if __name__ == '__main__':
    go_predict(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3])
    # go_predict(0, r"D:\project\humpWhale\data\humpback-whale-identification")
