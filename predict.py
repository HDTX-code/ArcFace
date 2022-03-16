from __init__ import *


def go_predict(a, data_root_path, save_root_path, model_30_url, model_15_30_url, model_10_14_url, model_7_9_url,
               model_5_6_url, model_0_4_url, th):
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dict_id_all_30, new_d_all_30, Feature_train_30, target_train_30, opt_30, model_30 = get_pre_need(data_root_path,
                                                                                                     save_root_path,
                                                                                                     model_30_url, 30,
                                                                                                     2000, 0, device)
    dict_id_all_15_30, new_d_all_15_30, Feature_train_15_30, target_train_15_30, opt_15_30, model_15_30 = get_pre_need(
        data_root_path,
        save_root_path,
        model_15_30_url, 15,
        30, 2, device)
    dict_id_all_10_14, new_d_all_10_14, Feature_train_10_14, target_train_10_14, opt_10_14, model_10_14 = get_pre_need(
        data_root_path,
        save_root_path,
        model_10_14_url, 10,
        14, 1, device)
    dict_id_all_7_9, new_d_all_7_9, Feature_train_7_9, target_train_7_9, opt_7_9, model_7_9 = get_pre_need(
        data_root_path,
        save_root_path,
        model_7_9_url, 7,
        9, 1, device)
    dict_id_all_5_6, new_d_all_5_6, Feature_train_5_6, target_train_5_6, opt_5_6, model_5_6 = get_pre_need(
        data_root_path,
        save_root_path,
        model_5_6_url, 5,
        6, 1, device)
    dict_id_all_0_4, new_d_all_0_4, Feature_train_0_4, target_train_0_4, opt_0_4, model_0_4 = get_pre_need(
        data_root_path,
        save_root_path,
        model_0_4_url, 0,
        4, 0, device)

    train_csv = pd.read_csv(os.path.join(data_root_path, "train.csv"))
    train_csv_id = train_csv['Id'].unique()
    dict_id = dict(zip(train_csv_id, range(len(train_csv_id))))
    new_d = {v: k for k, v in dict_id.items()}

    path_list = os.listdir(os.path.join(data_root_path, "test"))
    with tqdm(total=len(path_list), postfix=dict, file=sys.stdout) as pbar:
        l30 = 0
        l15 = 0
        l10 = 0
        l7 = 0
        l5 = 0
        l0 = 0
        submission = pd.DataFrame(columns=['Image', 'Id'])
        for item in range(len(path_list)):
            path = os.path.join(os.path.join(data_root_path, "test"), path_list[item])
            top4_30, top4_index_30 = get_pre(path, dict_id, dict_id_all_30, new_d_all_30, Feature_train_30,
                                             target_train_30, opt_30, model_30, device, float(th))
            l30 = l30 + len(top4_30)
            top4_15_30, top4_index_15_30 = get_pre(path, dict_id, dict_id_all_15_30, new_d_all_15_30,
                                                   Feature_train_15_30,
                                                   target_train_15_30, opt_15_30, model_15_30, device, float(th))
            l15 = l15 + len(top4_15_30)
            top4_10_14, top4_index_10_14 = get_pre(path, dict_id, dict_id_all_10_14, new_d_all_10_14,
                                                   Feature_train_10_14,
                                                   target_train_10_14, opt_10_14, model_10_14, device, float(th))
            l10 = l10 + len(top4_10_14)
            Top4 = np.concatenate((top4_30, top4_15_30, top4_10_14), axis=0)
            Top4_index = np.concatenate((top4_index_30, top4_index_15_30, top4_index_10_14), axis=0)
            if len(Top4) < 4:
                top4_7_9, top4_index_7_9 = get_pre(path, dict_id, dict_id_all_7_9, new_d_all_7_9,
                                                   Feature_train_7_9,
                                                   target_train_7_9, opt_7_9, model_7_9, device, float(th),
                                                   len(Top4) - 4)
                Top4 = np.concatenate((Top4, top4_7_9), axis=0)
                Top4_index = np.concatenate((Top4_index, top4_index_7_9), axis=0)
                l7 = l7 + len(top4_7_9)
                if len(Top4) < 4:
                    top4_5_6, top4_index_5_6 = get_pre(path, dict_id, dict_id_all_5_6, new_d_all_5_6,
                                                       Feature_train_5_6,
                                                       target_train_5_6, opt_5_6, model_5_6, device, float(th),
                                                       len(Top4) - 4)
                    Top4 = np.concatenate((Top4, top4_5_6), axis=0)
                    Top4_index = np.concatenate((Top4_index, top4_index_5_6), axis=0)
                    l5 = l5 + len(top4_index_5_6)
                    if len(Top4) < 4:
                        top4_0_4, top4_index_0_4 = get_pre(path, dict_id, dict_id_all_0_4, new_d_all_0_4,
                                                           Feature_train_0_4,
                                                           target_train_0_4, opt_0_4, model_0_4, device, -100,
                                                           len(Top4) - 4)
                        l0 = l0 + len(top4_0_4)
                        Top4 = np.concatenate((Top4, top4_0_4), axis=0)
                        Top4_index = np.concatenate((Top4_index, top4_index_0_4), axis=0)
            print(Top4)
            pbar.update(1)
            pbar.set_postfix(
                **{'model_30': l30, 'model_15-30': l15, 'model_10-14': l10, 'model_7-9': l7, 'model_5-6': l5,
                   'model_0-4': l0})
            submission.loc[item, "Image"] = item
            submission.loc[item, "Id"] = new_d[Top4_index[0]] + '\n' + new_d[Top4_index[1]] + '\n' + new_d[
                Top4_index[2]] + '\n' + new_d[Top4_index[3]] + '\n' + "new_whale"
        submission.to_csv(save_root_path, index=False)


if __name__ == '__main__':
    go_predict(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6],
               sys.argv[7], sys.argv[8], sys.argv[9])
    # go_predict(0, "D:\\project\\humpWhale\\data\\humpback-whale-identification",
    #            "D:\\project\\humpWhale\\data\\humpback-whale-identification",
    #            "D:\\edge\\gt30_model.pth",
    #            "D:\\edge\\resnet50_15-30_loss_ 0.7051018987383161score_ 0.9230769230769231.pth",
    #            "D:\\edge\\resnet50_10-14_loss_ 0.028108565703682278score_ 0.8936170212765957.pth",
    #            "D:\\edge\\resnet50_7-9_loss_ 1.95732459243463score_ 0.78125.pth",
    #            "D:\\edge\\resnet50_5-6_loss_ 2.8928167653638264score_ 0.5584415584415584.pth",
    #            "D:\\edge\\resnet50Sph-29loss_ 8.339440341671137score_ 0.pth",
    #            0.6)
