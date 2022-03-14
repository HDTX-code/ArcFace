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
        for item in path_list:
            top4_30, top4_index_30 = get_pre(item, dict_id, dict_id_all_30, new_d_all_30, Feature_train_30,
                                             target_train_30, opt_30, model_30, device, th)
            l30 = l30 + len(top4_30)
            top4_15_30, top4_index_15_30 = get_pre(item, dict_id, dict_id_all_15_30, new_d_all_15_30,
                                                   Feature_train_15_30,
                                                   target_train_15_30, opt_15_30, model_15_30, device, th)
            l15 = l15 + len(top4_15_30)
            top4_10_14, top4_index_10_14 = get_pre(item, dict_id, dict_id_all_10_14, new_d_all_10_14,
                                                   Feature_train_10_14,
                                                   target_train_10_14, opt_10_14, model_10_14, device, th)
            l10 = l10 + len(top4_10_14)
            Top4 = np.concatenate((top4_30, top4_15_30, top4_10_14), axis=1)
            Top4_index = np.concatenate((top4_index_30, top4_index_15_30, top4_index_10_14), axis=1)
            if len(Top4) < 4:
                top4_7_9, top4_index_7_9 = get_pre(item, dict_id, dict_id_all_7_9, new_d_all_7_9,
                                                   Feature_train_7_9,
                                                   target_train_7_9, opt_7_9, model_7_9, device, th, len(Top4) - 4)
                Top4 = np.concatenate((Top4, top4_7_9), axis=1)
                Top4_index = np.concatenate((Top4_index, top4_index_7_9), axis=1)
                l7 = l7 + len(top4_7_9)
                if len(Top4) < 4:
                    top4_5_6, top4_index_5_6 = get_pre(item, dict_id, dict_id_all_5_6, new_d_all_5_6,
                                                       Feature_train_5_6,
                                                       target_train_5_6, opt_5_6, model_5_6, device, -100, len(Top4) - 4)
                    Top4 = np.concatenate((Top4, top4_5_6), axis=1)
                    Top4_index = np.concatenate((Top4_index, top4_index_5_6), axis=1)
                    l5 = l5 + len(top4_index_5_6)

