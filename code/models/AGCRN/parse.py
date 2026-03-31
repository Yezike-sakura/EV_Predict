def add_model_specific_args(parser):
    parser.add_argument('--num_nodes', type=int, default=275, help='Number of nodes in UrbanEV')
    parser.add_argument('--embed_dim', type=int, default=10, help='Node embedding dimension')
    parser.add_argument('--rnn_units', type=int, default=64, help='Hidden units for AGCRN cell')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of AGCRN layers')
    parser.add_argument('--cheb_k', type=int, default=2, help='Chebyshev polynomial order')
    return parser