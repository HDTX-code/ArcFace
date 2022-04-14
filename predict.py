import copy

import numpy as np

from __init__ import *


def go_predict(args):
    with torch.no_grad():
        print(torch.cuda.is_available())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        path_list = os.listdir(args.data_test_path)
        # 建立test_dataloader的csv文件
        submission = pd.DataFrame(columns=['image', 'predictions'])
        for item in range(len(path_list)):
            submission.loc[item, "image"] = path_list[item]
        # 建立测试集地址字典
        dict_id_test = dict(zip(path_list, range(len(path_list))))
        new_d_test = {v: k for k, v in dict_id_test.items()}
        # 建立全局字典
        train_csv = pd.read_csv(args.data_csv_path)
        train_csv_id = train_csv['individual_id'].unique()
        dict_id_all = dict(zip(train_csv_id, range(len(train_csv_id))))
        new_d_all = {v: k for k, v in dict_id_all.items()}

        model, dict_id, Feature_train, target_train = get_pre_need(args.model_path, args.dict_id_path,
                                                                   args.train_csv_train_path,
                                                                   device, args.w, args.h,
                                                                   args.data_train_path, args.batch_size,
                                                                   args.num_workers, args.save_path, args.backbone,
                                                                   args.Feature_train_path, args.target_train_path)
        model.eval()

        test_dataset = TestDataset(submission, dict_id_test, args.data_test_path, args.w, args.h)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers)
        # Feature_test, target_test = get_feature(model, test_dataloader, device, 512)
        if args.Feature_test_path is None:
            Feature_test, target_test = get_feature(model, test_dataloader, device, 512)
            # target_test = target_test.cpu().detach().numpy()
            # Feature_test = Feature_test.cpu().detach().numpy()
            np.save(os.path.join(args.save_path, "Feature_test.npy"), Feature_test.cpu().detach().numpy())
            np.save(os.path.join(args.save_path, "target_test.npy"), target_test.cpu().detach().numpy())
        else:
            Feature_test = torch.from_numpy(np.load(args.Feature_test_path))
            target_test = torch.from_numpy(np.load(args.target_test_path))

        target_test = target_test.cpu().detach().numpy()

        if args.Top_all_path is None:
            # 获取各个总类中心点
            Feature_train_num = np.zeros([len(dict_id), 512])
            for item in range(len(dict_id)):
                Feature_train_num[item] = np.mean(Feature_train[target_train[:, 0] == item, :], axis=0)

            Top_all = np.zeros([len(path_list), 5])
            Top_index_all = np.zeros([len(path_list), 5])
            with tqdm(total=target_test.shape[0], postfix=dict) as pbar:
                for item in range(target_test.shape[0]):
                    # Top, Top_index = get_pre(Feature_test[item, :], Feature_train, target_train, dict_id,
                    #                          dict_id_all,
                    #                          4, device)
                    Top, Top_index = get_pre_num(Feature_test[item, :], Feature_train_num, dict_id, dict_id_all, 5,
                                                 device)
                    submission.loc[submission['image'] == new_d_test[target_test[item, 0]], "predictions"] = new_d_all[Top_index]
                    # Top_all[item, :] = copy.copy(Top)
                    # Top_index_all[item, :] = copy.copy(Top_index)
                    pbar.update(1)
                    pbar.set_postfix(**{'Top': Top})
            # np.save(os.path.join(args.save_path, "Top.npy"), Top_all)
            # np.save(os.path.join(args.save_path, "Top_index.npy"), Top_index_all)
        else:
            Top_all = np.load(args.Top_all_path)
            Top_index_all = np.load(args.Top_index_all_path)
        # with tqdm(total=Top_all.shape[0], postfix=dict) as pbar2:
        #     New_data = Top_all[np.argsort(Top_all[:, 0])[math.floor(0.12 * len(path_list))], 0]
        #     # Is_new = 'new_whale' if Top[0] < 0.75 else new_d_all[Top_index[4]]
        #     for item in range(len(path_list)):
        #         # submission.loc[
        #         #     submission[
        #         #         submission.image == new_d_test[target_test[item, 0]]].index.tolist(), "predictions"] = \
        #         #     new_d_all[Top_index_all[item, 0]] + ' ' + new_d_all[Top_index_all[item, 1]] + ' ' + new_d_all[
        #         #         Top_index_all[item, 2]] + ' ' + new_d_all[Top_index_all[item, 3]] + ' ' + new_d_all[
        #         #         Top_index_all[item, 4]]
        #         if Top_all[item, 0] <= New_data:
        #             submission.loc[
        #                 submission[
        #                     submission.image == new_d_test[target_test[item, 0]]].index.tolist(), "predictions"] = \
        #                 'new_individual' + ' ' + new_d_all[Top_index_all[item, 0]] + ' ' + new_d_all[
        #                     Top_index_all[item, 1]] + ' ' + new_d_all[Top_index_all[item, 2]] + ' ' + new_d_all[Top_index_all[item, 3]]
        #         else:
        #             submission.loc[
        #                 submission[
        #                     submission.image == new_d_test[target_test[item, 0]]].index.tolist(), "predictions"] = \
        #                 new_d_all[Top_index_all[item, 0]] + ' ' + 'new_individual' + ' ' + new_d_all[Top_index_all[item, 1]] + ' ' + new_d_all[
        #                     Top_index_all[item, 2]] + ' ' + new_d_all[Top_index_all[item, 3]]
        #         pbar2.update(1)
        #         pbar2.set_postfix(
        #             **{'Top': Top_all[item, 0], 'new': New_data})
    submission.to_csv(os.path.join(args.save_path, "submission.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练训练参数设置')
    parser.add_argument('--backbone', type=str, default='resnet50', help='特征网络选择，默认resnet50', required=True)
    parser.add_argument('--dict_id_path', type=str, help='字典路径', required=True)
    parser.add_argument('--model_path', type=str, help='模型路径', required=True)
    parser.add_argument('--train_csv_train_path', type=str, help='本次训练csv路径', default=None)
    parser.add_argument('--Feature_train_path', type=str, help='训练集特征矩阵路径', default=None)
    parser.add_argument('--target_train_path', type=str, help='训练集标签矩阵路径', default=None)
    parser.add_argument('--Feature_test_path', type=str, help='测试集特征矩阵路径', default=None)
    parser.add_argument('--target_test_path', type=str, help='测试集标签矩阵路径', default=None)
    parser.add_argument('--Top_all_path', type=str, help='测试集特征矩阵路径', default=None)
    parser.add_argument('--Top_index_all_path', type=str, help='测试集标签矩阵路径', default=None)
    parser.add_argument('--data_test_path', type=str, help='测试集路径', required=True)
    parser.add_argument('--data_train_path', type=str, help='训练集路径', default="../input/data-do-cut/All/All")
    parser.add_argument('--data_csv_path', type=str, help='训练集csv路径',
                        default=r'../input/happy-whale-and-dolphin/train.csv')
    parser.add_argument('--save_path', type=str, help='存储路径', default=r'./')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--w', type=int, help='训练图片宽度', default=224)
    parser.add_argument('--h', type=int, help='训练图片高度', default=224)
    args = parser.parse_args()

    go_predict(args)
    # # -------------------------------#
    # #   路径设置
    # # -------------------------------#
    # data_test_path = r'../input/unet-test/unet_test'
    # data_csv_path = r'../input/humpback-whale-identification/train.csv'
    # save_path = r'./'
    # path = r'../input/arc-epoth-3'
    # # -------------------------------#
    # #   dataloader设置
    # # -------------------------------#
    # w = 256
    # h = 256
    # num_workers = 2
    # batch_size = 128
    # # -------------------------------#
    # #   开始预测
    # # -------------------------------#
