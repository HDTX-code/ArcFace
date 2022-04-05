from __init__ import *


def go_predict_KNN(data_test_path, data_csv_path, save_path, path,
                   w, h, num_workers, batch_size, backbone, data_train_path, k,
                   backbone_1=None, backbone_2=None,
                   path_1=None, path_2=None):
    with torch.no_grad():
        print(torch.cuda.is_available())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 建立全局字典
        train_csv = pd.read_csv(data_csv_path)
        train_csv_id = train_csv['individual_id'].unique()
        dict_id_all = dict(zip(train_csv_id, range(len(train_csv_id))))
        new_d_all = {v: k for k, v in dict_id_all.items()}

        model, dict_id, Feature_train, target_train = get_pre_need(path, device, w, h,
                                                                   data_train_path, batch_size,
                                                                   num_workers, backbone)
        model.eval()
        if path_1 is not None:
            model_1, dict_id_1, Feature_train_1, target_train_1 = get_pre_need(path_1, device, w, h,
                                                                               data_train_path, batch_size,
                                                                               num_workers, backbone_1)
            model_1.eval()
        if path_2 is not None:
            model_2, dict_id_2, Feature_train_2, target_train_2 = get_pre_need(path_2, device, w, h,
                                                                               data_train_path, batch_size,
                                                                               num_workers, backbone_2)
            model_2.eval()

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

        Feature_test, target_test = get_feature(model, test_dataloader, device, 512)
        target_test = target_test.cpu().detach().numpy()
        # Feature_test = torch.from_numpy(np.load(r"C:\Users\12529\Desktop\1\Feature_test.npy"))
        # target_test = np.load(r"C:\Users\12529\Desktop\1\target_test.npy")

        if path_1 is not None:
            Feature_test_1, target_test_1 = get_feature(model_1, test_dataloader, device, 512)
        else:
            Feature_test_1 = None
        if path_2 is not None:
            Feature_test_2, target_test_2 = get_feature(model_2, test_dataloader, device, 512)
        else:
            Feature_test_2 = None

        KNN_by_iter(Feature_train, target_train,
                    Feature_test, target_test,
                    k, device, submission,
                    new_d_test, new_d_all, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练训练参数设置')
    parser.add_argument('--backbone', type=str, default='resnet50', help='特征网络选择，默认resnet50')
    parser.add_argument('--backbone_1', type=str, default='resnet101', help='特征网络选择，默认resnet101')
    parser.add_argument('--backbone_2', type=str, default='resnet152', help='特征网络选择，默认resnet152')
    parser.add_argument('--data_test_path', type=str, help='测试集路径', required=True)
    parser.add_argument('--data_train_path', type=str, help='训练集路径', default="../input/data-do-cut/All/All")
    parser.add_argument('--data_csv_path', type=str, help='训练csv路径',
                        default=r'../input/happy-whale-and-dolphin/train.csv')
    parser.add_argument('--save_path', type=str, help='存储路径', default=r'./')
    parser.add_argument('--path', type=str, help='模型及特征矩阵、字典存储路径', required=True)
    parser.add_argument('--path_1', type=str, help='模型及特征矩阵、字典存储路径', default=None)
    parser.add_argument('--path_2', type=str, help='模型及特征矩阵、字典存储路径', default=None)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--w', type=int, help='训练图片宽度', default=224)
    parser.add_argument('--h', type=int, help='训练图片高度', default=224)
    parser.add_argument('--k', type=int, help='KNN系数', default=200)
    args = parser.parse_args()

    go_predict_KNN(backbone=args.backbone,
                   save_path=args.save_path,
                   data_test_path=args.data_test_path,
                   data_csv_path=args.data_csv_path,
                   path=args.path,
                   num_workers=args.num_workers,
                   batch_size=args.batch_size,
                   data_train_path=args.data_train_path,
                   path_1=args.path_1,
                   path_2=args.path_2,
                   backbone_1=args.backbone_1,
                   backbone_2=args.backbone_2,
                   w=args.w,
                   h=args.h,
                   k=args.k)
