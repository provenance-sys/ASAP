import os
import sys
import torch
import pickle
import lmdb
import math
import copy
import random
import time
import heapq
from collections import deque, defaultdict
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch_geometric.utils import subgraph
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../.."))
sys.path.insert(0, project_root)
from src.utils import config
from src.dataloader.starting_dataloader import dataloadering
from src.dataloader.provdataloader import MyData
from src.models.eglad import EGLAD
from src.utils.provgraph import RedProvGraph, RawProvGraph
from src.utils.opreventmodel import OPREventModel


class EvalModel():
    def __init__(self, 
                args, 
                save_dir=None,
                ):
        self.args = args
        self.save_dir = save_dir
        self.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

        asap_save_path = os.path.join(config.artifact_dir, 'trained_asap', args.dataset, \
                                    "ASAP_MODEL" + '_pur' + str(args.pur_num) + '_mgn' + str(args.margin) + '_tc' + str(args.triplet_coef) \
                                    + '_ws' + str(args.sparsity_mask_coef) + '_we' + str(args.sparsity_ent_coef) \
                                    + '_dim' + str(args.hidden_dim) + '_ly' + str(args.num_hidden_layers) \
                                    + '_edim' + str(args.explainer_hidden_dim) + '_ely' + str(args.explainer_num_hidden_layers) \
                                    + '_lr' + str(args.lr) + '_R' + str(args.subgraph_radius))

        if os.path.exists(os.path.join(asap_save_path, 'ckpt', 'optimal_epoch.pkl')):
            optimal_epoch = pickle.load(open(os.path.join(asap_save_path, 'ckpt', 'optimal_epoch.pkl'), "rb"))
            self.model_path = os.path.join(asap_save_path, 'ckpt', 'model_epoch' + str(optimal_epoch) + '.pth')
        else:
            self.model_path = os.path.join(asap_save_path, 'ckpt', 'model_epoch.pth')

        self.attack_day_list = self.args.dataset_split[self.args.attack]
        
        self.res_save_path = os.path.join(self.save_dir, 'res')
        os.makedirs(self.res_save_path, exist_ok=True)

        self.groundtruth = None
        self.provgraph = None
        self.provgraph_nid2vid = {}
        self.provgraph_nid2content = {}
        self.provgraph_nid2type = {}
        self.fnid2file = {}
        self.fnid2score = {}
        self.attack_graph_node_nid = []
        self.attack_graph_node_vid = []
     
    
    def get_groundtruth(self):
        self.groundtruth = set()
        self.groundtruth_nid2type = {}
        self.groundtruth_nid2content = {}
        for attack_day in self.attack_day_list:
            groundtruth_file_path = os.path.join(os.path.dirname(__file__), "../../groundtruth/" + self.args.dataset + '_' + attack_day + '.txt')
            if os.path.exists(groundtruth_file_path):
                with open(groundtruth_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(maxsplit=2)
                        if len(parts) < 3:
                            continue
                        nid, node_type, content = parts
                        self.groundtruth.add(nid)
                        self.groundtruth_nid2type[nid] = node_type
                        self.groundtruth_nid2content[nid] = content
    
    def get_anomaly_node(self):
        if not self.args.is_retraining and os.listdir(self.res_save_path):
            print("anomaly file have been existed.")
            return
        dataloader = dataloadering(self.args)
        train_loader = dataloader['train']
        test_loader = dataloader[self.args.attack]

        sample_graph, _ = next(iter(train_loader))
        node_feat_dim = sample_graph.x.shape[1]
        edge_feat_dim = sample_graph.edge_attr.shape[1]

        asap_model = EGLAD(
            input_node_dim=node_feat_dim,
            input_edge_dim=edge_feat_dim,

            encoder_hidden_dim=self.args.hidden_dim,
            encoder_num_hidden_layers=self.args.num_hidden_layers,

            explainer_hidden_dim=self.args.explainer_hidden_dim,
            explainer_num_hidden_layers=self.args.explainer_num_hidden_layers,

            sparsity_mask_coef=self.args.sparsity_mask_coef,
            sparsity_ent_coef=self.args.sparsity_ent_coef,
            pur_num=self.args.pur_num,
            triplet_coef=self.args.triplet_coef,
        ).to(self.device)
        asap_model.load_state_dict(torch.load(self.model_path))
        asap_model.eval()

        all_ad_score = []
        for g, _ in tqdm(train_loader):
            torch.cuda.empty_cache()
            g = g.to(self.device)
            with torch.no_grad():
                ano_score = asap_model.get_anomaly_score(g)
                all_ad_score.append(ano_score.detach())
        ad_score = torch.cat(all_ad_score)
        torch.save(ad_score.cpu(), os.path.join(self.save_dir, "train_ad_score.pt"))
        
        subgraph_attribute_path = os.path.join(config.artifact_dir, 'dataloader', self.args.dataset, 'subgraph_attribute.lmdb')
        subgraph_attribute_env = lmdb.open(subgraph_attribute_path, readonly=True, lock=False)
        subgraph_attribute_txn = subgraph_attribute_env.begin()

        all_ad_true = []
        all_ad_score = []
        for g in tqdm(test_loader):
            torch.cuda.empty_cache()
            all_ad_true.append(g.y.cpu())
            g = g.to(self.device)
            with torch.no_grad():
                ano_score = asap_model.get_anomaly_score(g)
                batch_graph_emb = asap_model.get_graph_emb(g)
                node_imp = asap_model.get_node_imp(g)
                all_ad_score.append(ano_score.detach().cpu())

                central_nodes = g.graph_central_node
                graph_day = g.graph_day
                for graph_idx in range(len(central_nodes)):
                    
                    _central_node_idx = torch.tensor(central_nodes[graph_idx].item())  # 当前图的中心节点
                    _ano_score = ano_score[graph_idx].item()  # 当前图的异常分数
                    _graph_emb = batch_graph_emb[graph_idx, :]
                    
                    central_node_id = str(torch.tensor(g.node_id[_central_node_idx].item()))
                    attack_day = str(torch.tensor(graph_day[graph_idx].item()))
                    graph_group = attack_day + central_node_id
                    _central_node_nid = subgraph_attribute_txn.get((graph_group + central_node_id + 'nid').encode("utf-8")).decode('utf-8')

                    node_indices_in_graph = (g.batch == graph_idx).nonzero(as_tuple=True)[0]
                    anomaly_graph_node_list = set()
                    for node_idx in node_indices_in_graph:
                        node_id = str(torch.tensor(g.node_id[node_idx].item()))
                        _node_nid = subgraph_attribute_txn.get((graph_group + node_id + 'nid').encode("utf-8")).decode('utf-8')
                        _node_type = subgraph_attribute_txn.get((graph_group + node_id + 'type').encode("utf-8")).decode('utf-8')
                        _node_content = subgraph_attribute_txn.get((graph_group + node_id + 'content').encode("utf-8")).decode('utf-8')
                        _node_anomaly_score = node_imp[node_idx].item()
                        anomaly_graph_node_list.add((_node_nid, _node_anomaly_score, _node_content, _node_type))

                    anomaly_graphs = {}
                    anomaly_graphs['score'] = _ano_score
                    anomaly_graphs['emb'] = _graph_emb
                    anomaly_graphs['nodelist'] = anomaly_graph_node_list
                    anomaly_graphs['cnid'] = _central_node_nid
                    filename = f"{_ano_score:.4f}_{_central_node_nid.split(';')[0]}.pkl"
                    with open(os.path.join(self.res_save_path, filename), "wb") as f:
                        pickle.dump(anomaly_graphs, f)
        
    
    def get_sorted_anomaly_file_list(self):
        file_score_list = []
        for filename in os.listdir(self.res_save_path):
            if filename.endswith(".pkl"):
                try:
                    score = float(filename.split("_")[0])
                    central_node_fnid = filename.split("_")[1].split(".")[0]
                    file_score_list.append((score, filename))
                    self.fnid2file[central_node_fnid] = filename
                    self.fnid2score[central_node_fnid] = score
                except ValueError:
                    continue  
        file_score_list.sort(reverse=True, key=lambda x: x[0])
        return [filename for _, filename in file_score_list]

    def get_provgraph(self):
        self.provgraph = RedProvGraph(central_node_type=config.CENTRAL_NODE_TYPE)
        for attack_day in self.attack_day_list:
            _file = os.path.join(config.artifact_dir, 'reduce', self.args.dataset, attack_day + '.txt')
            with open(_file, 'r') as f:
                for line in tqdm(f):
                    oprem = OPREventModel()
                    oprem.update_from_loprem(line.strip().split('\t'))
                    self.provgraph.add_from_OPREM(oprem)
        
        for node, data in self.provgraph.nodes(data=True):
            node_contents = data['content'].split(';')
            node_nids = data['nid'].split(';')
            _is_equal = len(node_contents) == len(node_nids)
            for i, _nid in enumerate(node_nids):
                self.provgraph_nid2vid[_nid] = node
                self.provgraph_nid2type[_nid] = data['type']
                if _is_equal:
                    self.provgraph_nid2content[_nid] = node_contents[i]
                else:
                    self.provgraph_nid2content[_nid] = node_contents[0]
    
    def get_attack_nxsubgraph(self, subgraph_file_fnid, graph_thrd):
        with open(os.path.join(self.res_save_path, self.fnid2file[subgraph_file_fnid]), "rb") as f:
            subgraph = pickle.load(f)
        
        subgraph_vid_set = set()
        subgraph_reduced_nids_set = set()
        
        for _node_nid, _node_anomaly_score, _node_content, _node_type in subgraph['nodelist']:
            if _node_type == 'UnnamedPipeObject' or _node_type == 'MemoryObject':
                continue
            if float(_node_anomaly_score) < self.args.node_thed or len(_node_nid.split(';')) > 30:
                continue

            subgraph_reduced_nids_set.add(_node_nid)
            for _nid in _node_nid.split(';'):
                _vid = self.provgraph_nid2vid[_nid]
                subgraph_vid_set.add(_vid)
        attack_nxsubgraph = self.provgraph.subgraph(subgraph_vid_set)

        
        central_node_vid = self.provgraph_nid2vid[subgraph_file_fnid]
        if central_node_vid not in subgraph_vid_set:
            return subgraph_reduced_nids_set

        for _vid in subgraph_vid_set:
            if nx.has_path(attack_nxsubgraph, central_node_vid, _vid) or nx.has_path(attack_nxsubgraph, _vid, central_node_vid):
                continue
            path_nodes = None
            if nx.has_path(self.provgraph, central_node_vid, _vid):
                path_nodes = nx.shortest_path(self.provgraph, source=central_node_vid, target=_vid)
            if nx.has_path(self.provgraph, _vid, central_node_vid):
                path_nodes2 = nx.shortest_path(self.provgraph, source=_vid, target=central_node_vid)
                if path_nodes is None or len(path_nodes2) < len(path_nodes):
                    path_nodes = path_nodes2
            
            for path_nodes_vid in path_nodes:
                if len(self.provgraph.nodes[path_nodes_vid].get('nid').split(';')) > 30:
                    continue
                else:
                    subgraph_reduced_nids_set.add(self.provgraph.nodes[path_nodes_vid].get('nid'))
        
        return subgraph_reduced_nids_set
    
    def get_attack_node_list_from_poi(self, central_node_fnid, attack_graph_node_nid_set, attack_graph_node_reduced_nids_set, search_hops, graph_thrd):

        def bfs_traversal(_scan_fnid, depth):
            # print(_scan_fnid)
            if self.args.dataset == 'e3theia' and self.args.attack == 'browser_extension' and 'qt-opensource' in self.provgraph_nid2content[_scan_fnid.split(';')[0]]:
                return
            if depth >= search_hops or _scan_fnid not in self.fnid2file or self.fnid_is_scan_file[_scan_fnid]:
                return
            
            subgraph_reduced_nids_set = self.get_attack_nxsubgraph(_scan_fnid, graph_thrd)
            for _nids in subgraph_reduced_nids_set:
                for _nid in _nids.split(";"):
                    if _nid not in attack_graph_node_nid_set:
                        attack_graph_node_nid_set.add(_nid)
                        attack_graph_node_reduced_nids_set.add(_nids)

            self.fnid_is_scan_file[_scan_fnid] = True
            for _nids in subgraph_reduced_nids_set:
                for _nid in _nids.split(";"):
                    if _nid in self.fnid2score:
                        if self.fnid2score[_nid] >= graph_thrd:
                            bfs_traversal(_nids, depth=0)
                        else:
                            bfs_traversal(_nids, depth=depth+1)

        bfs_traversal(central_node_fnid, depth=0)
    
    def attack_reconstruction(self, search_hops=2):
        if self.provgraph is None:
            self.get_provgraph()
        if self.groundtruth is None:
            self.get_groundtruth()
        print(self.provgraph)
        
        train_ad_score = torch.load(os.path.join(self.save_dir, "train_ad_score.pt"))
        graph_thrd = torch.quantile(train_ad_score, self.args.graph_thed)

        anomaly_file_list = self.get_sorted_anomaly_file_list()
        self.fnid_is_scan_file = {}
        for filename in anomaly_file_list:
            central_node_fnid = filename.split("_")[1].split(".")[0]
            self.fnid_is_scan_file[central_node_fnid] = False
        
        attack_graph_node_nid_set = set()
        attack_graph_node_reduced_nids_set = set()
        rank = 0
        for filename in tqdm(anomaly_file_list):
            graph_anomaly_score = float(filename.split("_")[0])
            central_node_fnid = filename.split("_")[1].split(".")[0]

            if graph_anomaly_score < graph_thrd:
                break
            
            rank += 1
            if self.fnid_is_scan_file[central_node_fnid] == False:
                self.get_attack_node_list_from_poi(central_node_fnid, attack_graph_node_nid_set, attack_graph_node_reduced_nids_set, search_hops, graph_thrd)

            if len(attack_graph_node_reduced_nids_set) > self.args.invest_budget:
                break
        
        output_file = os.path.join(self.save_dir, "eval.txt")
        with open(output_file, "a", encoding="utf-8") as out:
            out.write(f'\n')
            out.write(f'POI_graph_nums: {rank}\n')
            out.write(f'\n')
        
        return attack_graph_node_nid_set, attack_graph_node_reduced_nids_set, rank
  
    def print_acc(self, attack_graph_node_nid_list, attack_graph_node_reduced_nids_set=None, search_hops=2):
        if self.provgraph is None:
            self.get_provgraph()
        if self.groundtruth is None:
            self.get_groundtruth()

        provgraph_node_nid = set()
        for node, attrs in self.provgraph.nodes(data=True):
            provgraph_node_nid.update(attrs['nid'].split(';'))

        prediction_attack_nid = set() 
        for node_nid in attack_graph_node_nid_list:
            for _nid in node_nid.split(';'):
                prediction_attack_nid.add(_nid)

        tp = len(prediction_attack_nid & self.groundtruth)
        fp = len(prediction_attack_nid - self.groundtruth)
        fn = len(self.groundtruth - prediction_attack_nid)
        tn = len(provgraph_node_nid) - tp - fp - fn

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) != 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

        numerator = (tp * tn) - (fp * fn)
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = numerator / denominator if denominator != 0 else 0

        output_file = os.path.join(self.save_dir, "eval.txt")
        with open(output_file, "a", encoding="utf-8") as out:
            out.write(f'\n')
            out.write(f'raw\n')
            out.write(f'tp: {tp}\n')
            out.write(f'tn: {tn}\n')
            out.write(f'fp: {fp}\n')
            out.write(f'fn: {fn}\n')
            out.write(f'accuracy: {accuracy}\n')
            out.write(f'precision: {precision}\n')
            out.write(f'recall: {recall}\n')
            out.write(f'f1: {f1}\n')
            out.write(f'fpr: {fpr}\n')
            out.write(f'mcc: {mcc}\n')
            if attack_graph_node_reduced_nids_set is not None:
                out.write(f'reduced num: {len(attack_graph_node_reduced_nids_set)}\n')
            if search_hops is not None:
                out.write(f'search hops: {search_hops}\n')
            out.write(f'node_thed: {self.args.node_thed}\n')
            out.write(f'\n')
        
        print(f'tp: {tp}')
        print(f'tn: {tn}')
        print(f'fp: {fp}')
        print(f'fn: {fn}')
        print(f'accuracy: {accuracy}')
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print(f'f1: {f1}')
        print(f'fpr: {fpr}')
        print(f'mcc: {mcc}')
        if attack_graph_node_reduced_nids_set is not None:
            print(f'reduced num: {len(attack_graph_node_reduced_nids_set)}')


    def get_uuid2score(self):
        if self.provgraph is None:
            self.get_provgraph()
        if self.groundtruth is None:
            self.get_groundtruth()
        print(self.provgraph)
        
        uuid2score = {}
        anomaly_file_list = self.get_sorted_anomaly_file_list()
        for filename in tqdm(anomaly_file_list):
            graph_anomaly_score = float(filename.split("_")[0])
            central_node_fnid = filename.split("_")[1].split(".")[0]
            _attack_graph = self.get_attack_nxsubgraph_new(central_node_fnid)

            for _, attrs in _attack_graph.nodes(data=True):
                _score = float(attrs['score']) * graph_anomaly_score
                for _nid in attrs['nid'].split(";"):
                    if _nid not in uuid2score:
                        uuid2score[_nid] = _score
                    elif uuid2score[_nid] < _score:
                        uuid2score[_nid] = _score

        with open(os.path.join(self.save_dir, "uuid2score.pt"), "wb") as uuid2score_f:
            pickle.dump(uuid2score, uuid2score_f)
