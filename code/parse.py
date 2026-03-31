import argparse


def get_global_parser():
    """
    全局共享参数总线。
    这里的参数对所有模型绝对统一，保证消融实验和对比实验的公平性。
    """
    parser = argparse.ArgumentParser(description="Go Spatio-temporal EV Charging Demand Prediction!")

    # 【硬件与全局设置】
    parser.add_argument('--device', type=int, default=0, help="CUDA.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")

    # 【核心时空控制变量 - 绝对公平基准】
    parser.add_argument('--seq_len', type=int, default=12, help="The sequence length of input data.")
    parser.add_argument('--pred_len', type=int, default=1, help="The length of prediction interval.")

    # 【全局训练控制】
    parser.add_argument('--bs', type=int, default=32, help="The batch size of fine-tuning.")
    parser.add_argument('--epoch', type=int, default=20, help="The max epoch of the training process.")
    # 在 【全局训练控制】 下方添加
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='None', help='adjust learning rate (e.g., type1, step, None)')

    # 【数据折叠与特征工程】
    parser.add_argument('--total_fold', type=int, default=6, help="The fold used for spliting data in cross-validation")
    parser.add_argument('--fold', type=int, default=0, help="The current fold number for training data")
    parser.add_argument('--add_feat', type=str, default='None',
                        help="Whether to use additional features for prediction")
    parser.add_argument('--feat', type=str, default='occ', help="Which feature to use for prediction")
    parser.add_argument('--pred_type', type=str, default='region', help="Prediction at node or regional level")

    # 【模型调度】
    parser.add_argument('--model', type=str, default='gcn', help="The used model (e.g., gcn, astgcn, adpstgcn)")
    parser.add_argument('--is_train', action='store_true', default=True)

    return parser