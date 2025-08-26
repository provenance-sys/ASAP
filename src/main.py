import os
import sys
import argparse
import yaml
import tracemalloc
import time

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../.."))
sys.path.insert(0, project_root)
from src.utils import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="e3cadets")
    parser.add_argument('-a', '--attack', type=str, default=None)

    parser.add_argument('--device', type=int, default=None)  # 1
    parser.add_argument('--train_batch_size', type=int, default=None)
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('-lr', '--lr', type=float, default=None)
    parser.add_argument('-R', '--subgraph_radius', type=int, default=None)
    parser.add_argument('-dim', '--hidden_dim', type=int, default=None)
    parser.add_argument('-ly', '--num_hidden_layers', type=int, default=None)
    parser.add_argument('-edim', '--explainer_hidden_dim', type=int, default=None)
    parser.add_argument('-ely', '--explainer_num_hidden_layers', type=int, default=None)

    parser.add_argument('--glad_epochs', type=int, default=None)
    parser.add_argument('--glad_save_epoch', type=int, default=None)

    parser.add_argument('--asap_epochs', type=int, default=None)
    parser.add_argument('--asap_save_epoch', type=int, default=None)
    parser.add_argument('-pur', '--pur_num', type=int, default=None)
    parser.add_argument('-mgn', '--margin', type=float, default=None)
    parser.add_argument('-tc', '--triplet_coef', type=float, default=None)
    parser.add_argument('-ws', '--sparsity_mask_coef', type=float, default=None)
    parser.add_argument('-we', '--sparsity_ent_coef', type=float, default=None)

    parser.add_argument('-gt', '--graph_thed', type=float, default=None)
    parser.add_argument('-nt', '--node_thed', type=float, default=None)
    
    args = parser.parse_args()

    yml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/" + args.dataset + ".yml"))
    with open(yml_path, "r") as f:
        parm_ymal = yaml.safe_load(f)
    for key, value in parm_ymal.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    all_dates = [date for dates in args.dataset_split.values() for date in dates]

    data_path = os.path.join(config.artifact_dir, 'dataloader', args.dataset)
    if not os.path.isdir(data_path) or not os.listdir(data_path):

        from src.parsers.starting_e3parse import parsering_darpae3
        parsering_darpae3(dataset_name=args.dataset)

        from src.reduction.starting_reduction import reducing_graph
        reducing_graph(dataset_name=args.dataset, all_datas=all_dates)

        from src.dataloader.starting_dataloader import dataloadering
        dataloadering(args=args)

    from src.train.starting_train_asap import training_asap
    training_asap(args=args)

    from src.eval.starting_eval import evaluating
    evaluating(args=args)

