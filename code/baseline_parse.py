def add_model_specific_args(parent_parser):
    """
    当前 baselines.py 内部各类传统/基础模型的专属参数。
    后续你把模型单独拆分到 models/GCN 等文件夹时，这个文件也可以跟着拆分。
    """
    parser = parent_parser.add_argument_group('Baselines_Config')

    # GCN & GCNLSTM 专属参数 (参考你 baselines.py 里 GCN 的默认超参)
    parser.add_argument('--gcn_hidden', type=int, default=32, help='GCN hidden dimensions')
    parser.add_argument('--gcn_layers', type=int, default=2, help='Number of GCN layers')

    # LSTM & GCNLSTM 专属参数
    parser.add_argument('--lstm_hidden_dim', type=int, default=256, help='LSTM hidden dimensions')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of LSTM layers')

    # ASTGCN 专属参数示例 (你可以根据需要添加)
    parser.add_argument('--nb_block', type=int, default=2, help='ASTGCN block count')
    parser.add_argument('--K', type=int, default=3, help='Chebyshev polynomial order')

    return parent_parser