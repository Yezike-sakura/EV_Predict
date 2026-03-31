# models/PDG2Seq/parse.py
def add_model_specific_args(parent_parser):
    parser = parent_parser.add_argument_group('PDG2Seq_Config')

    parser.add_argument('--input_dim', type=int, default=3, help='输入特征维度(1占有率+2时间特征)')
    parser.add_argument('--output_dim', type=int, default=1, help='输出维度')
    parser.add_argument('--rnn_units', type=int, default=64, help='GRU隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='RNN层数')
    parser.add_argument('--cheb_k', type=int, default=2, help='切比雪夫多项式阶数')
    parser.add_argument('--embed_dim', type=int, default=10, help='节点嵌入维度')
    parser.add_argument('--time_dim', type=int, default=10, help='时间嵌入维度')
    parser.add_argument('--use_day', type=bool, default=True, help='使用 day 嵌入')
    parser.add_argument('--use_week', type=bool, default=True, help='使用 week 嵌入')
    parser.add_argument('--horizon', type=int, default=1, help='预测步长, 会被全局 pred_len 覆盖')
    parser.add_argument('--lr_decay_step', type=int, default=2000, help='课程式学习衰减步数')
    parser.add_argument('--num_nodes', type=int, default=247, help='节点数量')

    return parent_parser