
import torch 
import pandas as pd 
import numpy as np

import baselines
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class CreateDataset(Dataset):
    def __init__(self,args, occ, extra_feat,device):  # adj
        lb = args.seq_len
        pt = args.pred_len
        self.pred_type = args.pred_type
        occ, label = create_rnn_data(occ, lb, pt)
        self.occ = torch.Tensor(occ)
        self.label = torch.Tensor(label)

        self.extra_feat = 'None'

        # if extra_feat != 'None':
        #     extra_feat, _ = create_rnn_data(extra_feat,lb,pt)     原代码

        # ------------------------------------------------修改1（消除Numpy警告）
        if isinstance(extra_feat, np.ndarray):  # 🎯 严谨的矩阵类型判断
            extra_feat, _ = create_rnn_data(extra_feat, lb, pt)
        # ------------------------------------------------修改1（消除Numpy警告）

            self.extra_feat = torch.Tensor(extra_feat)
        self.device = device

    def __len__(self):
        return len(self.occ)

    def __getitem__(self, idx):# occ: batch, seq, node
        output_occ = torch.transpose(self.occ[idx, :, :], 0, 1).to(self.device)
        output_label = self.label[idx, :].to(self.device)

        # if self.extra_feat != 'None':     原代码

        # ------------------------------------------------修改2（消除Numpy警告）
        if isinstance(self.extra_feat, torch.Tensor):  # 🎯 严谨的张量类型判断
        # ------------------------------------------------修改2（消除Numpy警告）

            output_extra_feat = torch.transpose(self.extra_feat[idx, :, :], 0, 1).to(self.device)
            return output_occ, output_label,output_extra_feat
        else:
            return output_occ, output_label


def create_loaders(train_occ, valid_occ, test_occ,train_extra_feat, valid_extra_feat, test_extra_feat, args, device):
    train_dataset = CreateDataset(args, train_occ, train_extra_feat, device)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True)

    valid_dataset = CreateDataset(args, valid_occ, valid_extra_feat, device)

    # 🎯 核心拆弹：把 len(valid_occ) 改为 args.bs
    valid_loader = DataLoader(valid_dataset, batch_size=args.bs, shuffle=False)

    test_dataset = CreateDataset(args, test_occ, test_extra_feat, device)

    # 🎯 核心拆弹：把 len(test_occ) 改为 args.bs
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)


    # valid_dataset = CreateDataset(args, valid_occ, valid_extra_feat, device)
    # valid_loader = DataLoader(valid_dataset, batch_size=len(valid_occ), shuffle=False)
    # 
    # test_dataset = CreateDataset(args, test_occ, test_extra_feat, device)
    # test_loader = DataLoader(test_dataset, batch_size=len(test_occ), shuffle=False)

    return train_loader, valid_loader, test_loader



def read_data(args):
    """
    Read and preprocess the dataset for model input.
    """

    # Load datasets
    inf = pd.read_csv('../data/inf.csv', header=0, index_col=None)
    occ = pd.read_csv('../data/occupancy.csv', header=0, index_col=0)
    duration = pd.read_csv('../data/duration.csv', header=0, index_col=0)
    volume = pd.read_csv('../data/volume.csv', header=0, index_col=0)
    e_price = pd.read_csv('../data/e_price.csv', index_col=0, header=0).values
    s_price = pd.read_csv('../data/s_price.csv', index_col=0, header=0).values
    adj = pd.read_csv('../data/adj.csv', header=0, index_col=None)
    adj.index = adj.columns

    time = pd.to_datetime(occ.index)

    feat = occ
    if args.feat == 'duration':
        feat = duration
    elif args.feat == 'volume':
        feat = volume

    # Normalize
    charge_count_dict = dict(zip(inf['TAZID'].astype(str), inf['charge_count']))
    for col in occ.columns:
        charge_count = charge_count_dict[col]
        occ[col] = occ[col] / charge_count

    price_scaler = MinMaxScaler(feature_range=(0, 1))
    e_price = price_scaler.fit_transform(e_price)
    s_price = price_scaler.fit_transform(s_price)

    # Load weather data
    weather = pd.read_csv(r'../data/weather_central.csv', header=0, index_col='time')

    extra_feat = 'None'
    if args.add_feat != 'None':
        extra_feat = np.zeros([occ.shape[0], occ.shape[1], 1])
        add_feat_list = args.add_feat.split('+')
        for add_feat in add_feat_list:
            if add_feat == 'all':
                extra_feat = np.concatenate([extra_feat, e_price[:, :, np.newaxis]], axis=2)
                extra_feat = np.concatenate([extra_feat, s_price[:, :, np.newaxis]], axis=2)
                extra_feat = np.concatenate([extra_feat,
                                             np.repeat(weather.values[:, np.newaxis, :], occ.shape[1], axis=1)], axis=2)
            elif add_feat == 'e':
                extra_feat = np.concatenate([extra_feat, e_price[:, :, np.newaxis]], axis=2)
            elif add_feat == 's':
                extra_feat = np.concatenate([extra_feat, s_price[:, :, np.newaxis]], axis=2)


            # # -------------------第一次修改------------------------------
            # elif add_feat == 'time':  # 🎯 【新增】生成通用的时空时间戳特征
            #     # 假设数据是5分钟粒度 (24小时 * 12 = 288个时间步)
            #     t_i_d = (time.hour * 12 + time.minute // 5).values
            #     d_i_w = time.dayofweek.values
            #     # 扩充维度对齐 occ [Time_len, Nodes, 1]
            #     t_i_d_feat = np.repeat(t_i_d[:, np.newaxis, np.newaxis], occ.shape[1], axis=1)
            #     d_i_w_feat = np.repeat(d_i_w[:, np.newaxis, np.newaxis], occ.shape[1], axis=1)
            #     # 拼接到 extra_feat 中
            #     extra_feat = np.concatenate([extra_feat, t_i_d_feat, d_i_w_feat], axis=2)

                # ------------------- 终极解耦修改 ------------------------------
            elif add_feat == 'time':
                # 🎯 【通用时间协议】输出最原始、信息量最全的原子时间特征
                # 任何模型都可以根据这 4 个基底，自行在 Adapter 内推演出自己想要的格式
                min_of_day = (time.hour * 60 + time.minute).values  # [0, 1439]
                d_i_w = time.dayofweek.values  # [0, 6]
                d_i_m = time.day.values  # [1, 31]
                m_i_y = time.month.values  # [1, 12]

                # 扩充维度对齐 occ [Time_len, Nodes, 1]
                mod_feat = np.repeat(min_of_day[:, np.newaxis, np.newaxis], occ.shape[1], axis=1)
                diw_feat = np.repeat(d_i_w[:, np.newaxis, np.newaxis], occ.shape[1], axis=1)
                dim_feat = np.repeat(d_i_m[:, np.newaxis, np.newaxis], occ.shape[1], axis=1)
                miy_feat = np.repeat(m_i_y[:, np.newaxis, np.newaxis], occ.shape[1], axis=1)

                # 拼接到 extra_feat 中 (此时 extra_feat 的最后一维会增加 4)
                extra_feat = np.concatenate([extra_feat, mod_feat, diw_feat, dim_feat, miy_feat], axis=2)

            else:
                extra_feat = np.concatenate([extra_feat,
                                             np.repeat(weather[add_feat].values[:, np.newaxis, np.newaxis], occ.shape[1], axis=1)], axis=2)
        extra_feat = extra_feat[:, :, 1:]

    return np.array(feat), np.array(adj), extra_feat, time


def set_seed(seed, flag):
    if flag:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def division(data, train_rate=0.7, valid_rate=0.2, test_rate=0.1):
    data_length = len(data)
    train_division_index = int(data_length * train_rate)
    valid_division_index = int(data_length * (train_rate + valid_rate))
    test_division_index = int(data_length * (1 - test_rate))
    train_data = data[:train_division_index]
    valid_data = data[train_division_index:valid_division_index]
    test_data = data[test_division_index:]
    return train_data, valid_data, test_data


def load_net(args, adj, device,occ):
    adj_dense  = torch.Tensor(adj).to(device)
    num_node = occ.shape[1] if args.pred_type =='region' else 1
    n_fea = 1
    if args.add_feat == 'all':
        n_fea = 8
    elif args.add_feat == 'None':
        n_fea = 1
    else:
        # 👇【核心修改】：精准对接通用时间特征的维度账本
        for feat_name in args.add_feat.split('+'):
            if feat_name == 'time':
                n_fea += 4  # 🎯 匹配 time 协议输出的 4 维原子特征
            else:
                n_fea += 1
    if args.model == 'lstm':
        model = baselines.Lstm(args.seq_len, n_fea, node=num_node).to(device)
    elif args.model == 'lo':
        model = baselines.Lo(args)
    elif args.model == 'ar':
        model = baselines.Ar(pred_len=args.pred_len,lags=args.seq_len,args=args)
    elif args.model == 'arima':
        model = baselines.Arima(pred_len=args.pred_len,p=args.seq_len,args=args)
    elif args.model == 'fcnn':
        model = baselines.Fcnn(n_fea, node=num_node, seq=args.seq_len).to(device)
    elif args.model == 'gcnlstm':
        model = baselines.Gcnlstm(args.seq_len,adj_dense=adj_dense,n_fea=n_fea, node=num_node,gcn_out=32, gcn_layers=1,lstm_hidden_dim=32, lstm_layers=1
                 ,hidden_dim=32).to(device)
    elif args.model == 'gcn':
        model = baselines.Gcn(args.seq_len,n_fea=n_fea, adj_dense=adj_dense,gcn_hidden=32,gcn_layers=1).to(device)
    elif args.model == 'astgcn':
        model = baselines.Astgcn(adj_dense=adj_dense,nb_block=1,in_channels=n_fea, K=1, nb_chev_filter=32, nb_time_filter=32, time_strides=1,num_for_predict=1,len_input=12,num_of_vertices=num_node).to(device)

    # 🎯 【新增】在这里注册并实例化我们的 PDG2Seq 适配器
    elif args.model.lower() == 'pdg2seq':               # 动态导入你的 adapter，防止其它模型报错

        from models.PDG2Seq.pdg2seq_adapter import UrbanEV_PDG2Seq
        # 🎯 强制覆盖原作者写死的“地雷”参数 (强行接管)
        # 1. 动态对齐真实节点数，防止 UrbanEV 切换预测目标时崩溃
        args.num_nodes = num_node
        # 2. 强制设为 1，因为时间特征已被抽走建图，RNN 真正吃的特征只剩 1 维
        args.input_dim = 1
        # 注意：这里我们给 adapter 传入 args，因为它里面包含了所有的超参数
        model = UrbanEV_PDG2Seq(args).to(device)

    elif args.model.lower() == 'agcrn':
        from models.AGCRN.AGCRN_adapter import AGCRN_Adapter
        # 1. 对齐空间流转：决定了自适应隐图(NAPL)的大小与端到端卷积的 Reshape
        args.num_nodes = num_node
        # 2. 释放特征约束：允许 AGCRN 的 RNN 单元满载吸收多变量外生特征
        args.input_dim = n_fea
        # 3. 锁死预测目标：明确告诉 CNN 预测器，最终吐出的目标指标只有 1 维 (Occupancy)
        args.output_dim = 1
        model = AGCRN_Adapter(args).to(device)

    elif args.model.lower() == 'gwnet':
        from models.GWNET.GWNET_adapter import Model as GWNET_Adapter
        # 1. 动态对齐真实节点数
        args.num_nodes = num_node
        # 2. 对接刚刚算出的精确特征维度账本 (n_fea=5)
        args.in_dim = n_fea
        # 3. 对接输出预测步长
        args.out_dim = args.pred_len
        # 挂载适配器
        model = GWNET_Adapter(args).to(device)

    elif args.model.lower() == 'dygraph_patchformer':
        from models.DyGraphPatchFormer.dygraph_adapter import DyGraphPatchFormerAdapter

        args.num_nodes = num_node
        model = DyGraphPatchFormerAdapter(args, adj_dense).to(device)

    return model
#注册新模型在此添加


def create_rnn_data(dataset, lookback, predict_time):
    x = []
    y = []
    for i in range(len(dataset) - lookback - predict_time):
        x.append(dataset[i:i + lookback])
        y.append(dataset[i + lookback + predict_time - 1])
    return np.array(x), np.array(y)


def metrics(test_pre, test_real,args):
    eps = 2e-2
    MAPE_test_real = test_real.copy()
    MAPE_test_pre = test_pre.copy()
    MAPE_test_real[np.where(MAPE_test_real <= eps)] = np.abs(MAPE_test_real[np.where(MAPE_test_real <= eps)]) + eps
    MAPE_test_pre[np.where(MAPE_test_real <= eps)] = np.abs(MAPE_test_pre[np.where(MAPE_test_real <= eps)]) + eps

    MAPE = mean_absolute_percentage_error(MAPE_test_real, MAPE_test_pre)
    MAE = mean_absolute_error(test_real, test_pre)
    MSE = mean_squared_error(test_real, test_pre)
    RMSE = np.sqrt(MSE)
    RAE = np.sum(abs(MAPE_test_pre - MAPE_test_real)) / np.sum(abs(np.mean(MAPE_test_real) - MAPE_test_real))

    print('MAPE: {}'.format(MAPE))
    print('MAE:{}'.format(MAE))
    print('MSE:{}'.format(MSE))
    print('RMSE:{}'.format(RMSE))
    print(('RAE:{}'.format(RAE)))
    output_list = [MSE, RMSE, MAPE, RAE, MAE]
    return output_list


def split_cv(args,time, feat,train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1,extra_feat='None'):
    """
    Split dataset based on time for time-series rolling cross-validation.
    """
    assert len(time) == len(feat)
    fold = args.fold
    month_list = list(time.month.unique())
    assert args.total_fold == len(month_list)
    fold_time = time.month.isin(month_list[0:fold]).sum()

    train_end = int(fold_time * train_ratio)
    valid_start = train_end
    valid_end = int(valid_start + fold_time * valid_ratio)
    test_start = valid_end
    test_end = int(fold_time)


    train_feat = feat[:train_end]
    valid_feat = feat[valid_start:valid_end]
    test_feat = feat[test_start:test_end]

    scaler = 'None'

    if args.pred_type == 'region':
        if args.feat != 'occ':
            scaler = StandardScaler()
            train_feat = scaler.fit_transform(train_feat)
            valid_feat = scaler.transform(valid_feat)
            test_feat = scaler.transform(test_feat)
    else:
        node_idx = int(args.pred_type)
        if args.feat != 'occ':
            scaler = StandardScaler()
            train_feat = scaler.fit_transform(train_feat[:,node_idx].reshape(-1,1))
            valid_feat = scaler.transform(valid_feat[:,node_idx].reshape(-1,1))
            test_feat = scaler.transform(test_feat[:,node_idx].reshape(-1,1))
        else:
            train_feat = train_feat[:, node_idx].reshape(-1, 1)
            valid_feat = valid_feat[:, node_idx].reshape(-1, 1)
            test_feat = test_feat[:, node_idx].reshape(-1, 1)

    train_extra_feat, valid_extra_feat, test_extra_feat = 'None','None','None'

    # if extra_feat != 'None':      原代码

    # ------------------------------------------------修改3（消除Numpy警告）
    if isinstance(extra_feat, np.ndarray):  # 🎯 严谨的矩阵类型判断
    # ------------------------------------------------修改3（消除Numpy警告）

        train_extra_feat = extra_feat[:train_end]
        valid_extra_feat = extra_feat[valid_start:valid_end]
        test_extra_feat = extra_feat[test_start:test_end]

    return train_feat, valid_feat, test_feat, train_extra_feat, valid_extra_feat, test_extra_feat,scaler
