from __init__ import *


def go_predict(a, data_root_path, save_path, path_0_2, path_3_11, path_12_1000):
    with torch.no_grad():
        opt = Config()
        print(torch.cuda.is_available())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_0_2, dict_id_0_2, Feature_train_0_2, target_train_0_2 = get_pre_need(path_0_2, device)
        model_0_2.eval()
        model_3_11, dict_id_3_11, Feature_train_3_11, target_train_3_11 = get_pre_need(path_3_11, device)
        model_3_11.eval()
        model_12_1000, dict_id_12_1000, Feature_train_12_1000, target_train_12_1000 = get_pre_need(path_12_1000, device)
        model_12_1000.eval()

        path_list = os.listdir(os.path.join(data_root_path, "test"))
        with tqdm(total=len(path_list), postfix=dict) as pbar:
            submission = pd.DataFrame(columns=['Image', 'Id'])
            for item in range(len(path_list)):
                path = os.path.join(os.path.join(data_root_path, "test"), path_list[item])
                Top_0_2, Top_index_0_2 = get_pre(path, model_0_2, Feature_train_0_2, target_train_0_2, dict_id_0_2, opt,
                                                 1, device)
                Top_3_11, Top_index_3_11 = get_pre(path, model_3_11, Feature_train_3_11, target_train_3_11, dict_id_3_11
                                                   , opt, 2, device)
                Top_12_1000, Top_index_12_1000 = get_pre(path, model_12_1000, Feature_train_12_1000,
                                                         target_train_12_1000, dict_id_12_1000, opt,
                                                         1, device)
                Top = np.concatenate((Top_0_2, Top_3_11, Top_12_1000), axis=0)
                submission.loc[item, "Image"] = path_list[item]
                submission.loc[item, "Id"] = Top_index_0_2[0] + '\n' + Top_index_3_11[0] + '\n' + Top_index_3_11[1] + '\n' + Top_index_12_1000[0] + '\n' + 'new_whale'
                pbar.update(1)
                pbar.set_postfix(
                    **{'Top4': Top})
        submission.to_csv(os.path.join(save_path, "submission.csv"), index=False)


if __name__ == '__main__':
    go_predict(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
