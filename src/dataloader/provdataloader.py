import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import os
import sys
import re
import random
from tqdm import tqdm
import copy
from zxcvbn import zxcvbn
import hashlib
import math
import string
import lmdb
import pickle
from hashid import HashID
from sklearn.feature_extraction import FeatureHasher

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../.."))
sys.path.insert(0, project_root)
from src.utils import config
from src.utils.provgraph import RedProvGraph
from src.utils.opreventmodel import OPREventModel


class MyData(Data):
    def __inc__(self, key, value, store):
        if key == 'graph_central_node' or key == 'imp_edge_index':
            return self.x.size(0)
        else:
            return super().__inc__(key, value, store)

class ProvDataset(Dataset):
    def __init__(self,
                dataset_name: str = None,
                splite_name: str = None,  # eg. 'train'
                splite_list: list = None,  # eg. ['2018-04-02', '2018-04-03']
                data_dir: str = None,
                save_dir: str = None,
                node_feature_generator = None,
                node_content_dim: str = 16,
                subgraph_radius: int = 3,
                ):
        self.dataset_name = dataset_name
        self.splite_name = splite_name
        self.splite_list = splite_list
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.node_feature_generator = node_feature_generator

        self.node_content_dim = node_content_dim
        self.node_type_dim = len(config.node_type[dataset_name])
        self.edge_type_dim = len(config.edge_type[dataset_name])

        self.subgraph_radius = subgraph_radius

        self.batch_save_num_data = 1000
        self.subgraph_attribute_save_path = os.path.join(save_dir, 'subgraph_attribute.lmdb')
        self.sample_save_path = os.path.join(save_dir, self.splite_name + '_data.lmdb')
        if 'train' in self.splite_name:
            self.neg_sample_save_path = os.path.join(save_dir, self.splite_name + '_neg_data.lmdb')
        
        self.event_types = config.edge_type[dataset_name]
        self.node_types = config.node_type[dataset_name]
        self.num_entries = None
        self.detector = HashID()
        
        super(ProvDataset, self).__init__(root=save_dir)
    
    @property
    def raw_file_names(self):
        raw_files = []
        for _file in self.splite_list:
            raw_files.append(os.path.join(self.data_dir, _file + '.txt'))
        return raw_files
    
    @property
    def processed_file_names(self):
        return [os.path.join(self.save_dir, self.splite_name + '_complete')]
    
    def download(self):
        print("Dataset does not exist.")
        sys.exit(1)
    
    def process(self):
        print("process..")
        self.log_file = open(os.path.join(self.save_dir, self.splite_name + '.log'), 'a')

        nx_graph_data_list = []
        unique_graphs = set()
        if 'train' in self.splite_name:
            neg_nx_graph_data_list = []
            neggeneration = NegGeneration(self.dataset_name, self.node_feature_generator, self.node_content_dim, self.edge_type_dim)
        
        for _daily_data in self.splite_list:
            groundtruth_file_path = os.path.join(os.path.dirname(__file__), "../../groundtruth/" + self.dataset_name + '_' + _daily_data + '.txt')
            self.groundtruth = set()
            if os.path.exists(groundtruth_file_path):
                with open(groundtruth_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(maxsplit=2)
                        if len(parts) < 3:
                            continue
                        nid, _, _ = parts
                        self.groundtruth.add(nid)

            _file = os.path.join(self.data_dir, _daily_data + '.txt')
            self.log_file.write("loading " + _file + '\n')
            provgraph = RedProvGraph(central_node_type=config.CENTRAL_NODE_TYPE)
            with open(_file, 'r') as f:
                for line in tqdm(f):
                    oprem = OPREventModel()
                    oprem.update_from_loprem(line.strip().split('\t'))
                    provgraph.add_from_OPREM(oprem)
            
            print("remove duplicate edges...")
            self.log_file.write("remove duplicate edges...\n")
            # provgraph.remove_redundant_event_open_edges()
            self.remove_duplicate_edges_by_type_redu_graph(provgraph)
            self.log_file.write("subgraph num:" + str(len(provgraph.get_central_nodes())) + '\n')

            # Computing node embedding
            print("Adding node id...")
            self.log_file.write("Adding node id...\n")
            for n in provgraph.nodes:
                _content = provgraph.nodes[n].get('content', 'None')
                _nid = provgraph.nodes[n].get('nid', 'None')
                provgraph.nodes[n]['id'] = self.string_to_md5_int(_nid)
                # provgraph.nodes[n]['feat'] = self.get_node_feature(_content)

                if 'train' not in self.splite_name and self.groundtruth is not None:
                    provgraph.nodes[n]['label'] = self.get_node_label(_nid)
                else:
                    provgraph.nodes[n]['label'] = torch.tensor([-1], dtype=torch.int)
            
            print("Get process-central subgraph...")
            self.log_file.write("Get process-central subgraph...\n")
            central_node_ids = set(provgraph.get_central_nodes())
            for central_node_id in tqdm(central_node_ids):

                central_node_subgraph = provgraph.find_reachable_subgraph_sel_node(central_node_id, hops=self.subgraph_radius)
                if len(central_node_subgraph.nodes) < 2:
                    continue

                unique_graph_id = provgraph.nodes[central_node_id].get('content', 'None') + str(len(central_node_subgraph.nodes)) + str(len(central_node_subgraph.edges))
                
                if 'train' in self.splite_name: #or 'e3clearscope' not in self.dataset_name:  # deduplication
                    if unique_graph_id in unique_graphs:
                        continue
                    else:
                        unique_graphs.add(unique_graph_id)
                    
                central_node_subgraph.graph['central_node'] = central_node_id
                self.remove_duplicate_nodes_by_content(central_node_subgraph)

                self.log_file.write('\n')
                self.log_file.write("graph:" + str(central_node_id) + " nodes:" + str(len(central_node_subgraph.nodes)) + " edges:" + str(len(central_node_subgraph.edges)) + '\n')

                central_node_subgraph.graph['label'] = provgraph.nodes[central_node_id]['label']
                nx_graph_data_list.append(central_node_subgraph)

                if 'train' in self.splite_name:
                    neggeneration.set_nx_subgraph(copy.deepcopy(central_node_subgraph))
                    neg_central_node_subgraph = neggeneration.get_neg_nx_subgraph(central_node_subgraph)
                    self.log_file.write("neg_graph:" + str(central_node_id) + " nodes:" + str(len(neg_central_node_subgraph.nodes)) + " edges:" + str(len(neg_central_node_subgraph.edges)) + '\n')
                    
                    neg_central_node_subgraph.graph['label'] = torch.tensor([1], dtype=torch.int)
                    neg_nx_graph_data_list.append(neg_central_node_subgraph)
                
                if len(nx_graph_data_list) >= self.batch_save_num_data:
                    self.save_nx_graph_data_list(nx_graph_data_list, self.sample_save_path, _daily_data)
                    self.save_subgraph_attribute(nx_graph_data_list, self.subgraph_attribute_save_path, _daily_data)
                    del nx_graph_data_list
                    nx_graph_data_list = []
                    if 'train' in self.splite_name:
                        self.save_nx_graph_data_list(neg_nx_graph_data_list, self.neg_sample_save_path, _daily_data)
                        del neg_nx_graph_data_list
                        neg_nx_graph_data_list = []
                
                if len(unique_graphs) >= self.batch_save_num_data:
                    unique_graphs = set()
                    
            self.save_nx_graph_data_list(nx_graph_data_list, self.sample_save_path, _daily_data)
            self.save_subgraph_attribute(nx_graph_data_list, self.subgraph_attribute_save_path, _daily_data)
            nx_graph_data_list = []
            unique_graphs = set()
            if 'train' in self.splite_name:
                self.save_nx_graph_data_list(neg_nx_graph_data_list, self.neg_sample_save_path, _daily_data)
                neg_nx_graph_data_list = []
            del provgraph
    
        open(os.path.join(self.save_dir, self.splite_name + '_complete'), 'w').close()
    
    def save_subgraph_attribute(self, graph_data_list, data_save_path, _daily_data):

        graph_attribute_env = lmdb.open(data_save_path, map_size=20*1024*1024*1024)
        with graph_attribute_env.begin(write=True) as txn:
            for _nx_graph in graph_data_list:
                central_node = _nx_graph.graph['central_node']
                central_node_id = str(_nx_graph.nodes[central_node].get('id')[0])
                graph_group = str(torch.tensor(int(_daily_data.replace("-", "")), dtype=torch.long)) + central_node_id
                for node, attrs in _nx_graph.nodes(data=True):
                    node_id = str(_nx_graph.nodes[node].get('id')[0])
                    txn.put((graph_group + node_id + 'nid').encode("utf-8"), attrs['nid'].encode("utf-8"))
                    txn.put((graph_group + node_id + 'type').encode("utf-8"), attrs['type'].encode("utf-8"))
                    txn.put((graph_group + node_id + 'content').encode("utf-8"), attrs['content'].encode("utf-8"))
        graph_attribute_env.close()
    
    def save_nx_graph_data_list(self, graph_data_list, data_save_path, _daily_data):
        data_save_lmdb = lmdb.open(data_save_path, map_size=20*1024*1024*1024)
        txn = data_save_lmdb.begin(write=True)
        try:
            for _nx_graph in graph_data_list:
                nodes = list(_nx_graph.nodes)
                edges = list(_nx_graph.edges)
                node_mapping = {node: i for i, node in enumerate(nodes)}
                
                node_id = torch.stack([_nx_graph.nodes[n]['id'] for n in nodes])
                imp_edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in 
                                            _nx_graph.graph['imp_edges']], dtype=torch.long).t().contiguous()
                graph_central_node = torch.tensor([node_mapping[_nx_graph.graph['central_node']]], dtype=torch.long)

                edge_index = torch.tensor([[node_mapping[edge[0]], node_mapping[edge[1]]] 
                                            for edge in edges], dtype=torch.long).t().contiguous()

                node_feat = torch.stack([_nx_graph.nodes[n]['feat'] for n in nodes])
                edge_feat = torch.stack([_nx_graph.edges[e]['feat'] for e in edges])
                
                node_labels = torch.stack([_nx_graph.nodes[n]['label'] for n in nodes])
                edge_labels = torch.stack([_nx_graph.edges[e]['label'] for e in edges])
                graph_label = _nx_graph.graph['label']

                graph_data = MyData(x=node_feat, 
                            edge_index=edge_index, 
                            edge_attr=edge_feat, 
                            y=graph_label,
                            node_labels=node_labels,
                            edge_labels=edge_labels, 

                            node_id=node_id,
                            imp_edge_index=imp_edge_index,
                            graph_central_node=graph_central_node,
                            graph_day=torch.tensor(int(_daily_data.replace("-", "")), dtype=torch.long)
                            )
                num_entries = txn.stat()['entries']
                txn.put(str(num_entries).encode(), pickle.dumps(graph_data))
            txn.commit()
        except Exception as e:
            txn.abort()
            raise e
        data_save_lmdb.close()
    
    def _add_reverse_edges_and_expand_edge_attr(self, graph_data):
        edge_index = graph_data.edge_index  # [2, num_edges]
        edge_attr = graph_data.edge_attr    # [num_edges, old_dim]
        edge_type_dim = self.edge_type_dim
        num_edges, old_dim = edge_attr.shape
        new_dim = old_dim + edge_type_dim

        forward_attr = torch.nn.functional.pad(edge_attr, (0, edge_type_dim), value=0)
        reverse_edge_index = edge_index[[1, 0], :]
        reverse_attr = torch.zeros_like(forward_attr)
        reverse_attr[:, :edge_type_dim] = edge_attr[:, -edge_type_dim:]
        if old_dim > 2 * edge_type_dim:
            reverse_attr[:, edge_type_dim:-edge_type_dim] = edge_attr[:, edge_type_dim:-edge_type_dim]
        reverse_attr[:, -edge_type_dim:] = edge_attr[:, :edge_type_dim]

        graph_data.edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        graph_data.edge_attr = torch.cat([forward_attr, reverse_attr], dim=0)

        graph_data.edge_labels = torch.cat([graph_data.edge_labels, graph_data.edge_labels], dim=0)

        return graph_data

    @staticmethod
    def remove_duplicate_edges_by_type_redu_graph(nx_graph: nx.MultiDiGraph):
        edges_to_remove = []
        node_pairs = set(copy.deepcopy(nx_graph.edges()))
        for u, v in node_pairs:
            edges = nx_graph[u][v]
            type_dict = {}
            for key in edges:
                edge_type = edges[key].get('type')
                if edge_type not in type_dict:
                    type_dict[edge_type] = key
                else:
                    if nx_graph[u][v][type_dict[edge_type]]['ts'] > nx_graph[u][v][key]['ts']:
                        nx_graph[u][v][type_dict[edge_type]]['ts'] = nx_graph[u][v][key]['ts']
                    edges_to_remove.append((u, v, key))
            if len(edges_to_remove) > 1000:
                nx_graph.remove_edges_from(edges_to_remove)
                del edges_to_remove
                edges_to_remove = []
        nx_graph.remove_edges_from(edges_to_remove)
    
    @staticmethod
    def is_hashs(s):
        s = s.lower()
        if re.fullmatch(r'[a-f0-9]{16}', s):
            return True
        elif re.fullmatch(r'[a-f0-9]{32}', s):
            return True
        elif re.fullmatch(r'[a-f0-9]{40}', s):
            return True
        elif re.fullmatch(r'[a-f0-9]{64}', s):
            return True
        elif re.fullmatch(r'[a-f0-9]{128}', s):
            return True
        else:
            return False

    def filter_content(self, ntype, ncontent):
        content_parts = set(ncontent.split(';'))
        content_dir_parts = set()
        if 'Net' in ntype:
            for _part in content_parts:
                content_dir_parts.add(max(_part.split(':'), key=len) if ":" in _part else _part)
            return content_dir_parts
        else:
            for _part in content_parts:
                _ps = re.split(r'[/:._\-]+', _part)
                content_dir_parts.update([re.sub(r'\d+$', '', _p) for _p in _ps if not _p.isdigit() and not ProvDataset.is_hashs(_p)])
                
            return content_dir_parts
    
    def remove_duplicate_nodes_by_content(self, nx_graph: nx.MultiDiGraph, hops=3):
        node_pre_hash = {}
        node_hashes = {}
        for node in nx_graph.nodes:
            _content = nx_graph.nodes[node].get('content', '')
            _type = nx_graph.nodes[node].get('type', '')
            node_hashes[node] = hash(frozenset(self.filter_content(_type, _content))) % 2 ** 31
            node_pre_hash[node] = node_hashes[node]
        
        for _ in range(hops):
            for u, v, key, data in nx_graph.edges(data=True, keys=True):
                edge_type = data['type'].encode()

                node_hashes[u] += (node_pre_hash[v] + int(hashlib.md5(edge_type).hexdigest(), 16)) % 2 ** 31
                node_hashes[v] += (node_pre_hash[u] + int(hashlib.md5(edge_type).hexdigest(), 16)) % 2 ** 31
            
            for node in nx_graph.nodes:
                node_hashes[node] = int(hashlib.md5(str(node_hashes[node]).encode()).hexdigest(), 16) % 2 ** 31
                node_pre_hash[node] = node_hashes[node]

        
        unique_hash = {}
        deleted_nodes = set()
        remove_pairs = []
        for node in nx_graph.nodes:
            if node == nx_graph.graph['central_node']:
                nx_graph.nodes[node]['feat'] = self.get_node_feature(nx_graph.nodes[node]['content'], nx_graph.nodes[node]['type'])
                continue
            node_hash = node_hashes[node]

            if node_hash not in unique_hash:
                unique_hash[node_hash] = [node]
            else:
                unique_hash[node_hash].append(node)
        
        for _, node_list in unique_hash.items():
            perserve_node = node_list[0]
            for remove_node in node_list[1:]:
                if nx_graph.nodes[perserve_node]['type'] != nx_graph.nodes[remove_node]['type']:
                    nx_graph.nodes[remove_node]['feat'] = self.get_node_feature(nx_graph.nodes[remove_node]['content'], nx_graph.nodes[remove_node]['type'])
                    continue
                nx_graph.nodes[perserve_node]['nid'] = nx_graph.nodes[perserve_node]['nid'] + ';' + nx_graph.nodes[remove_node]['nid']
                if nx_graph.nodes[perserve_node]['type'] not in config.CENTRAL_NODE_TYPE:
                    nx_graph.nodes[perserve_node]['content'] = nx_graph.nodes[perserve_node]['content'] + ';' + nx_graph.nodes[remove_node]['content']
                
                deleted_nodes.add(remove_node)
                remove_pairs.append((remove_node, perserve_node))
            
            nx_graph.nodes[perserve_node]['feat'] = self.get_node_feature(nx_graph.nodes[perserve_node]['content'], nx_graph.nodes[perserve_node]['type'])
        
        imp_edges = list(nx_graph.graph.get('imp_edges', []))
        for remove_node, perserve_node in remove_pairs:
            for neighbor in list(nx_graph.neighbors(remove_node)):
                if neighbor in deleted_nodes or nx_graph.has_edge(perserve_node, neighbor):
                    continue
                for edge_key in list(nx_graph[remove_node][neighbor].keys()):
                    nx_graph.add_edge(perserve_node, neighbor, **nx_graph[remove_node][neighbor][edge_key])
            
            for neighbor in list(nx_graph.predecessors(remove_node)):
                if neighbor in deleted_nodes or nx_graph.has_edge(neighbor, perserve_node):
                    continue
                for edge_key in list(nx_graph[neighbor][remove_node].keys()):
                    nx_graph.add_edge(neighbor, perserve_node, **nx_graph[neighbor][remove_node][edge_key])
            
            nx_graph.remove_node(remove_node)

            imp_edges = [(perserve_node if u == remove_node else u, perserve_node if v == remove_node else v) for u, v in imp_edges]

        nx_graph.graph['imp_edges'] = set(imp_edges)
        self.remove_duplicate_edges_by_type_redu_graph(nx_graph)

        for u, v, key in nx_graph.edges(keys=True):
            _type = nx_graph[u][v][key].get('type', 'None')
            _utype = nx_graph.nodes[u].get('type', 'None')
            _vtype = nx_graph.nodes[v].get('type', 'None')

            edge_one_hot_feat = self.get_edge_feature(_type, _utype, _vtype)
            nx_graph[u][v][key]['feat'] = edge_one_hot_feat.reshape(-1)

            u_label = nx_graph.nodes[u].get('label', 0).item()
            v_label = nx_graph.nodes[v].get('label', 0).item()
            nx_graph[u][v][key]['label'] = torch.tensor([1], dtype=torch.int) if u_label == 1 and v_label == 1 else torch.tensor([0], dtype=torch.int)
    
    
    @staticmethod
    def string_to_md5_int(originstr):
        originstr = originstr.encode("utf-8")
        signaturemd5 = hashlib.md5()
        signaturemd5.update(originstr)

        hex_value = signaturemd5.hexdigest()
        hex_value_64bit = hex_value[:16]  # 截取前 64 位
        int_value = int(hex_value_64bit, 16)
        int64_max = 9223372036854775807
        int_value = int_value % int64_max 
        return torch.tensor([int_value], dtype=torch.int64)

    def get_node_feature(self, ncontent: str, ntype: str) -> torch.Tensor:
        node_features = self.node_feature_generator.get_embedding(ncontent, ntype)
        node_encoding = [0] * len(self.node_types)
        if ntype in self.node_types:
            node_encoding[self.node_types.index(ntype)] = 1
        return torch.cat([torch.tensor(node_features, dtype=torch.float32),
                        torch.tensor(node_encoding, dtype=torch.float32)], dim=0).reshape(-1)
    
    def get_edge_feature(self, etype: str, srctype: str, dsttype: str) -> torch.Tensor:
        e_encoding = [0] * len(self.event_types)
        if etype in self.event_types:
            e_encoding[self.event_types.index(etype)] = 1
        
        return torch.tensor(e_encoding, dtype=torch.float).reshape(-1)
    
    def get_node_label(self, nid: str) -> torch.Tensor:
        nid_list = nid.split(';')
        label = 1 if any(n in self.groundtruth for n in nid_list) else 0
        return torch.tensor([label], dtype=torch.int)
    
    def get(self, idx):
        sample_save_lmdb = lmdb.open(self.sample_save_path, readonly=True, lock=False)
        txn = sample_save_lmdb.begin()
        graph_data = pickle.loads(txn.get(str(idx).encode()))
        sample_save_lmdb.close()
        
        if 'train' in self.splite_name:
            neg_sample_save_lmdb = lmdb.open(self.neg_sample_save_path, readonly=True, lock=False)
            neg_txn = neg_sample_save_lmdb.begin()  # 手动开始一个读事务
            neg_graph_data = pickle.loads(neg_txn.get(str(idx).encode()))
            neg_sample_save_lmdb.close()
            return graph_data, neg_graph_data
        
        return graph_data
    
    def len(self):
        if self.num_entries is None:
            sample_save_lmdb = lmdb.open(self.sample_save_path, readonly=True, lock=False)
            txn = sample_save_lmdb.begin()
            self.num_entries = txn.stat()['entries']
            sample_save_lmdb.close()
        return self.num_entries


class NegGeneration():
    def __init__(self, dataset_name, node_feature_generator, node_feat_dim, edge_feat_dim):
        self.nx_graph_list = []
        self.node_feature_generator = node_feature_generator
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.dataset_name = dataset_name

        self.event_types = config.edge_type[dataset_name]
        self.node_types = config.node_type[dataset_name]

    def set_nx_subgraph(self, nx_graph):
        if len(self.nx_graph_list) > 500:
            num_to_remove = 100
            indices_to_remove = random.sample(range(len(self.nx_graph_list)), num_to_remove)
            for index in sorted(indices_to_remove, reverse=True):
                del self.nx_graph_list[index]
        self.nx_graph_list.append(nx_graph)
        
    def get_neg_nx_subgraph(self, nx_graph):
        return self.path_splicing_node_feature_perturbation(nx_graph)

    def empty_perturbation(self, nx_graph):
        neg_graph = copy.deepcopy(nx_graph)
        nx.set_node_attributes(neg_graph, torch.tensor([0], dtype=torch.int), 'label')
        return neg_graph
    
    def path_splicing_node_feature_perturbation(self, nx_graph):
        if len(self.nx_graph_list) == 0:
            return self.empty_perturbation(nx_graph)
        
        nx_graph = copy.deepcopy(nx_graph)
        nx_central_node = nx_graph.graph['central_node']
        nx.set_node_attributes(nx_graph, torch.tensor([0], dtype=torch.int), 'label')
        nx_graph.nodes[nx_central_node]['label'] = torch.tensor([-1], dtype=torch.int)

        selected_graph = copy.deepcopy(random.choice(self.nx_graph_list))
        selected_central_node = selected_graph.graph['central_node']
        nx.set_node_attributes(selected_graph, torch.tensor([0], dtype=torch.int), 'label')
        selected_graph.nodes[selected_central_node]['label'] = torch.tensor([-1], dtype=torch.int)

        imp_edges = selected_graph.graph.get('imp_edges', selected_graph.edges)
        num_perturbations = math.ceil(len(selected_graph.nodes) * (random.random() / 2 + 0.5))
        num_perturbations = num_perturbations if num_perturbations >= 5 else len(selected_graph.nodes)

        perturbed_nodes = set()
        perturbed_nodes.add(selected_central_node)
        
        def perturb_feat(node):
            if random.random() < 0.8:
                new_feat = self.node_feature_generator.get_random_word_embedding()
                node_features = torch.tensor(new_feat, dtype=torch.float32)

                node_encoding = [0] * len(self.node_types)
                ntype = selected_graph.nodes[node]['type']
                if ntype in self.node_types:
                    node_encoding[self.node_types.index(ntype)] = 1
                selected_graph.nodes[node]['feat'] =  torch.cat([torch.tensor(node_features, dtype=torch.float32),
                                                    torch.tensor(node_encoding, dtype=torch.float32)], dim=0).reshape(-1)

        def trace_back_to_central(node):
            path = []
            current_node = node
            while current_node != selected_central_node and current_node not in perturbed_nodes:
                path.append(current_node)
                predecessors = [u for u, v in imp_edges if v == current_node]
                if predecessors:
                    current_node = random.choice(predecessors)
                else:
                    break

            for n in reversed(path):
                if len(perturbed_nodes) >= num_perturbations:
                    break
                if n not in perturbed_nodes:
                    perturb_feat(n)
                    perturbed_nodes.add(n)
                    selected_graph.nodes[n]['label'] = torch.tensor([1], dtype=torch.int)

        while len(perturbed_nodes) < num_perturbations:
            unvisited_nodes = set(selected_graph.nodes) - perturbed_nodes
            if not unvisited_nodes:
                break
            selected_node = random.choice(list(unvisited_nodes))
            trace_back_to_central(selected_node)

        # Delete nodes in selected_graph that are not being visited
        nodes_to_remove = [n for n, data in selected_graph.nodes(data=True) if data['label'] == torch.tensor([0], dtype=torch.int)]
        selected_graph.remove_nodes_from(nodes_to_remove)

        # Replace the name and attributes of the center node of selected_graph with the nx_graph
        if selected_central_node in selected_graph.nodes:
            selected_graph = nx.relabel_nodes(selected_graph, {selected_central_node: nx_central_node})
            selected_graph.nodes[nx_central_node].update(nx_graph.nodes[nx_central_node])
        
        # Updata imp_edges
        selected_graph.graph['imp_edges'] = self.replace_nxgraph_imp_values(selected_graph.graph['imp_edges'], selected_central_node, nx_central_node)
        selected_graph.graph['imp_edges'] = self.filter_edges_by_graph_nodes(selected_graph, selected_graph.graph['imp_edges'])
        nx_graph.graph['imp_edges'].update(selected_graph.graph['imp_edges'])

        self.perturb_edge_one_hot(selected_graph)

        # Splice selected_graph to nx_graph
        nx_graph = nx.compose(selected_graph, nx_graph)
        return nx_graph
    
    @staticmethod
    def replace_nxgraph_imp_values(sets, str1, str2):
        return {(u if u != str1 else str2, v if v != str1 else str2) for u, v in sets}
    
    @staticmethod
    def filter_edges_by_graph_nodes(nx_graph, edge_set):
        valid_nodes = set(nx_graph.nodes)
        filtered_set = {(u, v) for u, v in edge_set if u in valid_nodes and v in valid_nodes}
        return filtered_set
    
    def perturb_edge_one_hot(self, nx_graph, random_prob=0.2):
        for u, v, key in nx_graph.edges(keys=True):
            if random.random() < random_prob:
                nx_graph[u][v][key]['feat'][:self.edge_feat_dim] = 0
                random_pos = torch.randint(0, self.edge_feat_dim, (1,)).item()
                nx_graph[u][v][key]['feat'][random_pos] = 1
    
