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

    if not args.is_retraining and os.path.exists(os.path.join(eval_save_path, "asap_detection_nid_list.pkl")):
        attack_graph_node_nid_list = pickle.load(open(os.path.join(eval_save_path, "asap_detection_nid_list.pkl"), "rb"))
    else:
        attack_graph_node_nid_list = eval_model.attack_reconstruction()
        pickle.dump(attack_graph_node_nid_list, open(os.path.join(eval_save_path, "asap_detection_nid_list.pkl"), "wb"))

    if len(attack_graph_node_nid_list) > 100:
        attack_graph_node_nid_list_200 = eval_model.print_reduced_acc(attack_graph_node_nid_list)
        eval_model.print_acc(attack_graph_node_nid_list_200)
        exit()

    if not args.is_retraining and os.path.exists(os.path.join(eval_save_path, "asap_traversed_nid_list.pkl")):
        attack_graph_node_nid_list = pickle.load(open(os.path.join(eval_save_path, "asap_traversed_nid_list.pkl"), "rb"))
        eval_model.plt_attack_graph_node_nid_list(attack_graph_node_nid_list, "asap")
    else:
        attack_graph_node_nid_list = eval_model.attack_reconstruction_traversed(attack_graph_node_nid_list, "asap")
        pickle.dump(attack_graph_node_nid_list, open(os.path.join(eval_save_path, "asap_traversed_nid_list.pkl"), "wb"))
    attack_graph_node_nid_list_200 = eval_model.print_reduced_acc(attack_graph_node_nid_list, 100)
    eval_model.print_acc(attack_graph_node_nid_list_200, 100)

if __name__ == '__main__':
    pass
