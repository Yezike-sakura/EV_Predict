def add_model_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("DyGraphPatchFormer_Config")
    parser.add_argument("--graph_hidden_dim", type=int, default=64, help="Hidden dim for graph generator")
    parser.add_argument("--lambda_init", type=float, default=0.1, help="Initial lambda in hybrid graph")
    parser.add_argument(
        "--graph_norm",
        type=str,
        default="row",
        choices=["row", "sym", "none"],
        help="Normalization for hybrid adjacency",
    )
    parser.add_argument("--sym_graph", action="store_true", default=False, help="Enforce symmetric dynamic graph")
    parser.add_argument(
        "--graph_nonneg_mode",
        type=str,
        default="relu",
        choices=["relu", "softplus"],
        help="Non-negative projection for adjacency",
    )
    return parent_parser

