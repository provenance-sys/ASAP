import torch
import os
import sys
import argparse
import pickle

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../.."))
sys.path.insert(0, project_root)
from src.utils import config
from src.eval.evaluations import EvalModel


def evaluating(args):
    evaluation_dir = os.path.join(config.artifact_dir, 'eval_res', args.dataset)
    os.makedirs(evaluation_dir, exist_ok=True)

    eval_save_path = os.path.join(evaluation_dir, "ASAP_MODEL_" + str(args.attack) + '_pur' + str(args.pur_num) + '_mgn' + str(args.margin) + '_tc' + str(args.triplet_coef) \
                            + '_ws' + str(args.sparsity_mask_coef) + '_we' + str(args.sparsity_ent_coef) \
                            + '_dim' + str(args.hidden_dim) + '_ly' + str(args.num_hidden_layers) \
                            + '_edim' + str(args.explainer_hidden_dim) + '_ely' + str(args.explainer_num_hidden_layers) \
                            + '_lr' + str(args.lr) + '_R' + str(args.subgraph_radius) \
                            + '_gt' + str(args.graph_thed) + '_nt' + str(args.node_thed) + '_eph' + str(args.asap_save_epoch))
    os.makedirs(eval_save_path, exist_ok=True)

    eval_model = EvalModel(args=args, save_dir=eval_save_path)
    eval_model.get_anomaly_node()
    attack_graph_node_nid_list, attack_graph_node_reduced_nids_set, _ = eval_model.attack_reconstruction()
    eval_model.print_acc(attack_graph_node_nid_list, attack_graph_node_reduced_nids_set)


if __name__ == '__main__':
    pass