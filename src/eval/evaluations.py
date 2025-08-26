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


def get_anomaly_rank(x_node_true, x_node_score, log_file):
    sorted_indices = np.argsort(-x_node_score)
    ranks = np.where(np.isin(sorted_indices, np.where(x_node_true == 1)[0]))[0] + 1
    for rank, idx in zip(ranks, sorted_indices[ranks - 1]):
        log_file.write(f"Rank: {rank}, Node Index: {idx}, Anomaly Score: {x_node_score[idx]}\n")
        # print(f"Rank: {rank}, Node Index: {idx}, Anomaly Score: {x_node_score[idx]}")
    log_file.write('\n')

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
        self.attack_graph_node_nid_set = set()  # 辅助集合，用于快速查找
        self.attack_graph = RedProvGraph(central_node_type=config.CENTRAL_NODE_TYPE)
        
    
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
        
        '''
        central_node_nid: emb: _graph_emb
                        score: _ano_score
                        subgraph: {(_node_nid, _node_anomaly_score, _node_content, _node_type), ...}
                        cnid: _central_node_nid
        '''
        
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
        
        all_ad_score = torch.cat(all_ad_score)
        ad_true = torch.cat(all_ad_true)
        ad_score = all_ad_score
        # rank_log_file = open(os.path.join(self.save_dir, "rank.txt"), 'a', buffering=1)
        # get_anomaly_rank(ad_true.numpy(), ad_score.numpy(), rank_log_file)
    
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
    
    def get_attack_nxsubgraph(self, subgraph_file_fnid):
        with open(os.path.join(self.res_save_path, self.fnid2file[subgraph_file_fnid]), "rb") as f:
            subgraph = pickle.load(f)

        subgraph_nid2vidlist = {}
        subgraph_vid_list = set()
        for _node_nid, _node_anomaly_score, _node_content, _node_type in subgraph['nodelist']:
            if _node_type == 'UnnamedPipeObject' or _node_type == 'MemoryObject':
                continue
            subgraph_nid2vidlist[_node_nid] = set()
            for _nid in _node_nid.split(';'):
                _vid = self.provgraph_nid2vid[_nid]
                subgraph_vid_list.add(_vid)
                subgraph_nid2vidlist[_node_nid].add(_vid)
                self.provgraph.nodes[_vid]['score'] = float(_node_anomaly_score)
        attack_nxsubgraph = self.provgraph.subgraph(subgraph_vid_list) #.copy()

        #################################待定，是否要保证图的连通性？#################################
        attack_nxsubgraph_cvid = self.provgraph_nid2vid[subgraph['cnid'].split(';')[0]]
        if attack_nxsubgraph.nodes[attack_nxsubgraph_cvid]['score'] >= self.args.node_thed:
            for _node_vid, attrs in attack_nxsubgraph.nodes(data=True):
                if attrs['score'] >= self.args.node_thed:
                    path_nodes = None
                    if nx.has_path(attack_nxsubgraph, _node_vid, attack_nxsubgraph_cvid):
                        path_nodes = nx.shortest_path(attack_nxsubgraph, source=_node_vid, target=attack_nxsubgraph_cvid)
                    if nx.has_path(attack_nxsubgraph, attack_nxsubgraph_cvid, _node_vid):
                        path_nodes2 = nx.shortest_path(attack_nxsubgraph, source=attack_nxsubgraph_cvid, target=_node_vid)
                        if path_nodes is None or len(path_nodes2) < len(path_nodes):
                            path_nodes = path_nodes2
                    if path_nodes is not None:
                        for _vid in path_nodes:
                            if attack_nxsubgraph.nodes[_vid]['score'] < self.args.node_thed:
                                self.provgraph.nodes[_vid]['score'] = attrs['score']
        #################################待定，是否要保证图的连通性？#################################

        perserve_node_vids = set()
        perserve_node_vid2vidlist = {}
        perserve_node_vid2score = {}
        for _node_nid, vid_set in subgraph_nid2vidlist.items():
            vid_list = list(vid_set)
            perserve_node_vids.add(vid_list[0])
            perserve_node_vid2vidlist[vid_list[0]] = vid_list
            perserve_node_vid2score[vid_list[0]] = self.provgraph.nodes[vid_list[0]]['score']
        
        save_graph = self.provgraph.subgraph(perserve_node_vids).copy()
        save_oprem = []
        for edges_u, edges_v, _key, data in list(save_graph.edges(keys=True, data=True)):
            u_attrs = save_graph.nodes[edges_u]
            v_attrs = save_graph.nodes[edges_v]
            loprem = [edges_u, \
                    ";".join(str(attack_nxsubgraph.nodes[u].get('nid', '')) for u in perserve_node_vid2vidlist[edges_u]), \
                    u_attrs.get('type', None), \
                    ";".join(str(attack_nxsubgraph.nodes[u].get('content', '')) for u in perserve_node_vid2vidlist[edges_u]), \
                    u_attrs.get('flag', None), \
                    edges_v, \
                    ";".join(str(attack_nxsubgraph.nodes[v].get('nid', '')) for v in perserve_node_vid2vidlist[edges_v]), \
                    v_attrs.get('type', None), \
                    ";".join(str(attack_nxsubgraph.nodes[v].get('content', '')) for v in perserve_node_vid2vidlist[edges_v]), \
                    v_attrs.get('flag', None), \
                    data.get('type', None), data.get('ts', None), data.get('te', None)
                    ]
            oprem = OPREventModel()
            oprem.update_from_loprem(loprem)
            save_oprem.append(copy.deepcopy(oprem))
        return attack_nxsubgraph, subgraph['cnid'], save_oprem, perserve_node_vid2score
    
    def get_attack_node_list_from_poi(self, central_node_fnid, _skip=False):
        initial_anomaly_subject_fnid_dict = {}  # fnid: graph_score
        initial_unanomaly_subject_fnid_dict = {}  # fnid: graph_score
        fnid_is_forward_scan_file = {}
        fnid_is_backward_scan_file = {}
        fnid2nxsubgraph = {}  # cnid: graph
        fnid2cnid = {}
        fnid2saveoprem = {}
        fnid2savescore = {}

        def bfs_traversal(_scan_fnid, is_forward, is_backward, depth, gs=1.0):
            if self.args.dataset == 'e3theia' and self.args.attack == 'browser_extension' and self.provgraph_nid2content[_scan_fnid] == "/home/admin/qt-opensource-linux-x64-5.10.1.run":
                return
            if depth >= 5 or _scan_fnid not in self.fnid2file:
                return
            if depth > 0 and self.fnid_is_scan_file[_scan_fnid]:
                return
            
            if is_forward and is_backward:
                if fnid_is_forward_scan_file.get(_scan_fnid) == True and fnid_is_backward_scan_file.get(_scan_fnid) == True:
                    return
            elif is_forward:
                if fnid_is_forward_scan_file.get(_scan_fnid) == True:
                    return
            elif is_backward:
                if fnid_is_backward_scan_file.get(_scan_fnid) == True:
                    return
                
            if is_forward:
                if fnid_is_forward_scan_file.get(_scan_fnid) == True:
                    is_forward = False
                else:
                    fnid_is_forward_scan_file[_scan_fnid] = True
            if is_backward:
                if fnid_is_backward_scan_file.get(_scan_fnid) == True:
                    is_backward = False
                else:
                    fnid_is_backward_scan_file[_scan_fnid] = True
            
            
        
            if _scan_fnid not in fnid2nxsubgraph:
                attack_nxsubgraph, attack_nxsubgraph_cnid, save_oprem, perserve_node_vid2score = self.get_attack_nxsubgraph(_scan_fnid)
                fnid2nxsubgraph[_scan_fnid] = attack_nxsubgraph
                fnid2cnid[_scan_fnid] = attack_nxsubgraph_cnid
                fnid2saveoprem[_scan_fnid] = save_oprem
                fnid2savescore[_scan_fnid] = perserve_node_vid2score
            else:
                attack_nxsubgraph = fnid2nxsubgraph[_scan_fnid]
                attack_nxsubgraph_cnid = fnid2cnid[_scan_fnid]
                save_oprem = fnid2saveoprem[_scan_fnid]
                perserve_node_vid2score = fnid2savescore[_scan_fnid]
            attack_nxsubgraph_cvid = self.provgraph_nid2vid[attack_nxsubgraph_cnid.split(';')[0]]

            if float(perserve_node_vid2score[attack_nxsubgraph_cvid]) < self.args.node_thed or gs < self.args.node_thed:
                initial_unanomaly_subject_fnid_dict[_scan_fnid] = self.fnid2score[_scan_fnid]
                return

            if _scan_fnid not in initial_anomaly_subject_fnid_dict:
                initial_anomaly_subject_fnid_dict[_scan_fnid] = self.fnid2score[_scan_fnid]

            # print(attack_nxsubgraph_cnid, attack_nxsubgraph.nodes[attack_nxsubgraph_cvid]['content'], depth)

            forward_depth_dict = self.ts_monotonic_bfs(graph=attack_nxsubgraph, source=attack_nxsubgraph_cvid, max_depth=self.args.subgraph_radius, increasing=True)
            reversed_attack_nxsubgrap = attack_nxsubgraph.reverse()
            backward_depth_dict = self.ts_monotonic_bfs(graph=reversed_attack_nxsubgrap, source=attack_nxsubgraph_cvid, max_depth=self.args.subgraph_radius, increasing=False)

            next_forward_subject_fnid = {}
            next_backward_subject_fnid = {}
            for _forward_node, _ in forward_depth_dict.items():
                if attack_nxsubgraph.nodes[_forward_node]['type'] in config.CENTRAL_NODE_TYPE: # and attack_nxsubgraph.nodes[_forward_node]['score'] >= self.args.node_thed:
                    for _nide_fnid in attack_nxsubgraph.nodes[_forward_node]['nid'].split(';'):
                        if _nide_fnid in self.fnid2score and _nide_fnid not in fnid_is_forward_scan_file:
                            next_forward_subject_fnid[_nide_fnid] = attack_nxsubgraph.nodes[_forward_node]['score']
            for _backward_node, _ in backward_depth_dict.items():
                if attack_nxsubgraph.nodes[_backward_node]['type'] in config.CENTRAL_NODE_TYPE: # and attack_nxsubgraph.nodes[_backward_node]['score'] >= self.args.node_thed:
                    for _nide_fnid in attack_nxsubgraph.nodes[_backward_node]['nid'].split(';'):
                        if _nide_fnid in self.fnid2score and _nide_fnid not in fnid_is_backward_scan_file:
                            next_backward_subject_fnid[_nide_fnid] = attack_nxsubgraph.nodes[_backward_node]['score']

            # print("")
            # print("forward...")
            # for _subject_fnid in next_forward_subject_fnid:
            #     print(self.provgraph.nodes[self.provgraph_nid2vid[_subject_fnid]]['content'])
            # print("")
            # print("backward...")
            # for _subject_fnid in next_backward_subject_fnid:
            #     print(self.provgraph.nodes[self.provgraph_nid2vid[_subject_fnid]]['content'])
            # print("")

            if len(next_forward_subject_fnid) <= 5:
                for _subject_fnid in next_forward_subject_fnid:
                    bfs_traversal(_subject_fnid, is_forward=True, is_backward=False, depth=depth+1, gs=next_forward_subject_fnid[_subject_fnid])
            if len(next_backward_subject_fnid) <= 5:
                for _subject_fnid in next_backward_subject_fnid:
                    bfs_traversal(_subject_fnid, is_forward=False, is_backward=True, depth=depth+1, gs=next_backward_subject_fnid[_subject_fnid])
        bfs_traversal(central_node_fnid, is_forward=True, is_backward=True, depth=0)

        for _central_node_fnid, _ in sorted(initial_anomaly_subject_fnid_dict.items(), key=lambda item: item[1], reverse=True):
            if self.fnid_is_scan_file[_central_node_fnid]:
                continue
            self.fnid_is_scan_file[_central_node_fnid] = True
            self.attack_graph.add_from_graph(fnid2saveoprem[_central_node_fnid], fnid2savescore[_central_node_fnid])
        
        for _central_node_fnid, _ in initial_unanomaly_subject_fnid_dict.items():
            if self.fnid_is_scan_file[_central_node_fnid]:
                continue
            self.fnid_is_scan_file[_central_node_fnid] = True
            self.attack_graph.add_from_graph(fnid2saveoprem[_central_node_fnid], {k: 0. for k in fnid2savescore[_central_node_fnid]})

    
    def attack_reconstruction(self):
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
        
        _rank = 1
        attack_graph_node_nid_list = []
        attack_graph_node_nid_set = set()
        self.attack_graph = RedProvGraph(central_node_type=config.CENTRAL_NODE_TYPE)
        for filename in tqdm(anomaly_file_list):
            graph_anomaly_score = float(filename.split("_")[0])
            central_node_fnid = filename.split("_")[1].split(".")[0]

            if graph_anomaly_score < graph_thrd:
                break
            # print("rank:", _rank)
            _rank += 1

            self.get_attack_node_list_from_poi(central_node_fnid, True)

            for _, attrs in sorted(
                self.attack_graph.nodes(data=True),
                key=lambda item: float(item[1].get('score', 0)),
                reverse=True
            ):
                if attrs['type'] == 'UnnamedPipeObject' or attrs['type'] == 'MemoryObject':
                    continue
                if float(attrs['score']) < self.args.node_thed:
                    break
                if attrs['nid'] not in attack_graph_node_nid_set:
                    attack_graph_node_nid_set.add(attrs['nid'])
                    attack_graph_node_nid_list.append(attrs['nid'])


        return attack_graph_node_nid_list
    
    def attack_reconstruction_traversed(self, attack_graph_node_nid_list, save_name):
        attack_graph_node_nid_set = set(attack_graph_node_nid_list)

        for _, attrs in sorted(
                self.attack_graph.nodes(data=True),
                key=lambda item: float(item[1].get('score', 0)),
                reverse=True
            ):
            if attrs['type'] == 'UnnamedPipeObject' or attrs['type'] == 'MemoryObject':
                continue
            if float(attrs['score']) >= self.args.node_thed:
                continue
            if attrs['nid'] not in attack_graph_node_nid_set:
                attack_graph_node_nid_set.add(attrs['nid'])
                attack_graph_node_nid_list.append(attrs['nid'])
        
        train_ad_score = torch.load(os.path.join(self.save_dir, "train_ad_score.pt"))
        graph_thrd = torch.quantile(train_ad_score, self.args.graph_thed)

        self.attack_graph = RedProvGraph(central_node_type=config.CENTRAL_NODE_TYPE)
        anomaly_file_list = self.get_sorted_anomaly_file_list()
        _rank = 1
        for filename in tqdm(anomaly_file_list):
            graph_anomaly_score = float(filename.split("_")[0])
            central_node_fnid = filename.split("_")[1].split(".")[0]
            # print("rank:", _rank)
            _rank += 1

            if self.fnid_is_scan_file[central_node_fnid]:
                continue

            # print("strp 1")
            self.get_attack_node_list_from_poi(central_node_fnid)
            # print("strp 2")

            for _, attrs in sorted(
                self.attack_graph.nodes(data=True),
                key=lambda item: float(item[1].get('score', 0)),
                reverse=True
            ):
                if attrs['type'] == 'UnnamedPipeObject' or attrs['type'] == 'MemoryObject':
                    continue
                if float(attrs['score']) < self.args.node_thed:
                    break
                if attrs['nid'] not in attack_graph_node_nid_set:
                    attack_graph_node_nid_set.add(attrs['nid'])
                    attack_graph_node_nid_list.append(attrs['nid'])
            
            for _, attrs in sorted(
                self.attack_graph.nodes(data=True),
                key=lambda item: float(item[1].get('score', 0)),
                reverse=True
            ):
                if attrs['type'] == 'UnnamedPipeObject' or attrs['type'] == 'MemoryObject':
                    continue
                if float(attrs['score']) >= self.args.node_thed:
                    continue
                if attrs['nid'] not in attack_graph_node_nid_set:
                    attack_graph_node_nid_set.add(attrs['nid'])
                    attack_graph_node_nid_list.append(attrs['nid'])
            
            self.attack_graph = RedProvGraph(central_node_type=config.CENTRAL_NODE_TYPE)
        self.plt_attack_graph_node_nid_list(attack_graph_node_nid_list, save_name)
        return attack_graph_node_nid_list

    def plt_attack_graph_node_nid_list(self, attack_graph_node_nid_list, save_name):
        if self.groundtruth is None:
            self.get_groundtruth()
        output_file = os.path.join(self.save_dir, f"traversal_groundtruth_{save_name}.txt")
        with open(output_file, "w", encoding="utf-8") as out:
            TP_count = 0
            y_percent = [0]
            x_count = [0]
            total_groundtruth = len(self.groundtruth)
            is_scaned_pos = set()
            percent = 0
            idx = 1
            for _, (_node_nid) in enumerate(attack_graph_node_nid_list, 1):
                is_scaned = True
                for _nnid in _node_nid.split(';'):
                    if _nnid not in is_scaned_pos:
                        is_scaned = False
                if is_scaned:
                    continue

                for _nnid in _node_nid.split(';'):
                    if _nnid in self.groundtruth and _nnid not in is_scaned_pos:
                        is_scaned_pos.add(_nnid)
                        TP_count += 1
                        percent = TP_count / total_groundtruth * 100 if total_groundtruth > 0 else 0
                        y_percent.append(percent)
                        x_count.append(idx)
                idx += 1

        plt.figure()
        plt.plot(x_count, y_percent, marker='o')
        plt.xscale('log')  # 横坐标指数级增长
        plt.xlabel('Number of traversed nodes')
        plt.ylabel('Recall (%)')
        plt.title('Traversed Nodes vs. Recall')
        plt.grid(True)
        plt.xlim(left=1)  # x轴从0开始
        plt.savefig(os.path.join(self.save_dir, f"recall_vs_traversed_nodes_groundtruth_{save_name}.png"))
        plt.close()

    @staticmethod
    def ts_monotonic_bfs(graph, source, max_depth, increasing=True):
        visited = {}  # node -> depth
        queue = deque()
        queue.append((source, 0, None))  # (当前节点, 当前深度, 上一个时间戳)

        while queue:
            current_node, depth, last_ts = queue.popleft()
            if depth > max_depth:
                continue

            if current_node not in visited:
                visited[current_node] = depth
            else:
                continue

            for succ in graph.successors(current_node):
                for _, edge_attr in graph.get_edge_data(current_node, succ).items():
                    ts = edge_attr.get('ts')
                    if ts is None:
                        continue

                    if last_ts is None or \
                        (increasing and ts > last_ts) or \
                        (not increasing and ts < last_ts):
                            queue.append((succ, depth + 1, ts))
                            break
        return visited

    def print_acc(self, attack_graph_node_nid_list, max_nnum=200):
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
            out.write(f'\n')

            print("")
            print(f'raw')
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

    def print_reduced_acc(self, attack_graph_node_nid_list, max_nnum=200):
        if self.groundtruth is None:
            self.get_groundtruth()

        attack_graph_node_nid_list_200 = []
        tp = 0
        fp = 0
        prediction_attack_nid = set() 
        for node_nid in attack_graph_node_nid_list:
            is_tp = False
            nnum = 0
            for _nid in node_nid.split(';'):
                if _nid not in prediction_attack_nid:
                    nnum += 1
                    if _nid in self.groundtruth:
                        is_tp = True
                    prediction_attack_nid.add(_nid)
            if nnum == 0:
                continue
            else:
                attack_graph_node_nid_list_200.append(node_nid)
            if is_tp:
                tp += 1
            else:
                fp += 1
            
            if len(attack_graph_node_nid_list_200) >= max_nnum:
                break

        output_file = os.path.join(self.save_dir, "eval_reduced.txt")
        with open(output_file, "a", encoding="utf-8") as out:
            out.write(f'\n')
            out.write(f'reduced\n')
            out.write(f'tp: {tp}\n')
            out.write(f'fp: {fp}\n')
            out.write(f'\n')

            print(f'reduced')
            print(f'tp: {tp}')
            print(f'fp: {fp}')
        
        return attack_graph_node_nid_list_200
