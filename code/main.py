import utils
import torch
# 【修改 1】：引入新的 Parser 构建器，而不是直接 import parse_args
from parse import get_global_parser

import train
import numpy as np
from utils import split_cv, create_loaders

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 【架构核心】：动态参数缝合逻辑 (Plug-and-Play)
    # ---------------------------------------------------------
    # 1. 拿取全局骨架
    parser = get_global_parser()

    # 2. 预解析，偷看一眼用户想跑什么模型 (--model)
    temp_args, _ = parser.parse_known_args()

    # 3. 根据模型名称，动态插拔专属参数 U 盘
    # 这里兼容现有的 baseline 模型
    if temp_args.model in ['gcn', 'lstm', 'gcnlstm', 'astgcn', 'fcnn', 'lo', 'ar', 'arima']:
        try:
            from baseline_parse import add_model_specific_args

            parser = add_model_specific_args(parser)
        except ImportError:
            pass
    # 未来你新增的模型，只需在这里加 elif 即可：
    elif temp_args.model.lower() == 'pdg2seq':
        try:
            from models.PDG2Seq.parse import add_model_specific_args

            parser = add_model_specific_args(parser)
        except ImportError as e:
            print(f"Error loading PDG2Seq args: {e}")

    elif temp_args.model.lower() == 'agcrn':
        try:
            from models.AGCRN.parse import add_model_specific_args

            parser = add_model_specific_args(parser)
        except ImportError as e:
            print(f"Error loading AGCRN args: {e}")

    elif temp_args.model.lower() == 'gwnet':
        try:
            # 如果你有专属的参数解析文件，就动态加载；没有则静默跳过
            from models.GWNET.parse import add_model_specific_args

            parser = add_model_specific_args(parser)
        except ImportError:
            pass
    elif temp_args.model.lower() == 'dygraph_patchformer':
        try:
            from models.DyGraphPatchFormer.parse import add_model_specific_args

            parser = add_model_specific_args(parser)
        except ImportError as e:
            print(f"Error loading DyGraphPatchFormer args: {e}")

    # 4. 最终合并生成完美参数表
    args = parser.parse_args()
    # ---------------------------------------------------------

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    utils.set_seed(seed=args.seed, flag=True)
    feat, adj, extra_feat, time = utils.read_data(args)

    print(
        f"Running {args.model} with feat={args.feat}, pre_l={args.pred_len}, fold={args.fold}, add_feat={args.add_feat}, pred_type(node)={args.pred_type}")

    # Initialize and train model
    net = utils.load_net(args, np.array(adj), device, feat)

    train_feat, valid_feat, test_feat, train_extra_feat, valid_extra_feat, test_extra_feat, scaler = split_cv(args,
                                                                                                              time,
                                                                                                              feat,
                                                                                                              TRAIN_RATIO,
                                                                                                              VAL_RATIO,
                                                                                                              TEST_RATIO,
                                                                                                              extra_feat)
    train_loader, valid_loader, test_loader = create_loaders(train_feat, valid_feat, test_feat,
                                                             train_extra_feat, valid_extra_feat,
                                                             test_extra_feat,
                                                             args, device)
    if args.model == 'lo' or args.model == 'ar' or args.model == 'arima':
        optim = None
        loss_func =None
        args.is_train = False
        args.stat_model = True
        train_valid_feat = np.vstack((train_feat, valid_feat,test_feat[:args.seq_len+args.pred_len,:]))
        test_loader = [train_valid_feat,test_feat[args.pred_len+args.seq_len:,:]]
    else:
        optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=0.00001)
        # optim = torch.optim.Adam(net.parameters(), weight_decay=0.00001)
        args.stat_model = False

        # loss_func = torch.nn.MSELoss()          #MSE计算loss
        loss_func = torch.nn.L1Loss()         #MAE计算loss
        # loss_func = torch.nn.SmoothL1Loss()   #Smooth L1 Loss

        if args.is_train:
            train.training(args, net, optim, loss_func, train_loader, valid_loader, args.fold)

    train.test(args, test_loader, feat, net, scaler)
