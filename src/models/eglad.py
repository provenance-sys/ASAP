import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool, Set2Set
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GIN
from torch_scatter import scatter_max, scatter, scatter_min
import torch.multiprocessing as mp
import networkx as nx
import random
import copy
import math
import os
import sys

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../.."))
sys.path.insert(0, project_root)
from src.models.glad import GLAD, ExplainerGNN
from src.utils import config
from src.utils.datasaver import OpremSaveStrategy


class EGLAD(nn.Module):
    def __init__(self, 
                input_node_dim, 
                input_edge_dim, 

                encoder_hidden_dim,
                encoder_num_hidden_layers,

                explainer_hidden_dim,
                explainer_num_hidden_layers,

                sparsity_mask_coef = 0.01,
                sparsity_ent_coef = 0.01,
                pur_num=10,
                triplet_coef = 0.5,
                margin=0.2,

                ):
        super(EGLAD, self).__init__()

        self.explainer = ExplainerGNN(input_node_dim, input_edge_dim, explainer_hidden_dim, explainer_num_hidden_layers, pur_num)
        self.glad = GLAD(input_node_dim, input_edge_dim, encoder_hidden_dim, encoder_num_hidden_layers)
        
        self.pool = global_add_pool
        
        # self.imp_aggregation = MaxProductPathLayer(att_factor=1/2)
        self.pos_att_aggregation = PosAttPathLayer()
        self.neg_att_aggregation = NegAttPathLayer()

        self.sparsity_mask_coef = sparsity_mask_coef
        self.sparsity_ent_coef = sparsity_ent_coef
        self.pur_num = pur_num
        self.triplet_coef = triplet_coef
        self.margin = margin

    def forward(self, g, neg_g):
        node_att = self.get_node_att(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, batch=g.batch, imp_edge_index=g.imp_edge_index, graph_central_node=g.graph_central_node)

        sparsity_loss = self.sparsity_loss(node_att)
        rec_loss = self.rec_loss(g, node_att)
    
        if self.pur_num == 0:
            return rec_loss.mean() * 10 + sparsity_loss.mean()
        
        triplet_loss = self.triplet_loss(neg_g)
        return rec_loss.mean() * 10 + sparsity_loss.mean() + triplet_loss.mean() * self.triplet_coef * 10
    
    def rec_loss(self, g, node_att):
        rec_loss = self.glad(g=g, node_imp=node_att)
        return rec_loss
    
    def sparsity_loss(self, edge_mask, eps=1e-6):
        sparsity = 0.
        edge_mask = torch.clamp(edge_mask, min=eps, max=1-eps)
        ent = -edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        sparsity += self.sparsity_mask_coef * edge_mask
        sparsity += self.sparsity_ent_coef * ent
        return sparsity.reshape(-1)

    def triplet_loss(self, neg_g):
        triplets = self.generate_triplets(neg_g)
        loss = 0.
        for pos_pur_mask, neg_pur_mask, all_pur_mask in triplets:
            pos_pur_graph_emb = self.get_neg_graph_emb(neg_g, pos_pur_mask)
            neg_pur_graph_emb = self.get_neg_graph_emb(neg_g, neg_pur_mask)
            all_pur_graph_emb = self.get_neg_graph_emb(neg_g, all_pur_mask)
            
            _weight = (self.get_neg_graph_anomaly_score(neg_g, all_pur_mask).detach()) * (self.get_neg_graph_anomaly_score(neg_g, neg_pur_mask).detach())
            loss += self.triplet_loss_cosine(all_pur_graph_emb, neg_pur_graph_emb, pos_pur_graph_emb) * _weight.clamp(min=1e-10)
        return loss

    def get_neg_graph_emb(self, neg_g, mask):
        node_att = self.explainer(x=neg_g.x, edge_index=neg_g.edge_index, edge_attr=neg_g.edge_attr, batch=neg_g.batch, imp_edge_index=neg_g.imp_edge_index, graph_central_node=neg_g.graph_central_node, node_imp=mask)
        node_att = node_att.reshape(-1) * mask.reshape(-1)
        graph_emb = self.glad.get_graph_emb(x=neg_g.x, edge_index=neg_g.edge_index, batch=neg_g.batch, node_imp=node_att)
        return graph_emb
    
    def get_neg_graph_anomaly_score(self, neg_g, mask):
        node_att = self.explainer(x=neg_g.x, edge_index=neg_g.edge_index, edge_attr=neg_g.edge_attr, batch=neg_g.batch, imp_edge_index=neg_g.imp_edge_index, graph_central_node=neg_g.graph_central_node, node_imp=mask)
        node_att = node_att.reshape(-1) * mask.reshape(-1)
        anomaly_score = self.glad.get_anomaly_score(neg_g, node_imp=node_att)
        return anomaly_score

    def triplet_loss_cosine(self, anchor, positive, negative):
        sim_ap = F.cosine_similarity(anchor, positive, dim=1)  # anchor-positive
        sim_an = F.cosine_similarity(anchor, negative, dim=1)  # anchor-negative
        dist_ap = 1 - sim_ap
        dist_an = 1 - sim_an
        loss = F.relu(dist_ap - dist_an + self.margin)
        return loss

    def get_node_att(self, x, edge_index, edge_attr, batch, imp_edge_index, graph_central_node):
        node_att = self.explainer(x, edge_index, edge_attr, batch, imp_edge_index, graph_central_node)
        return node_att
    
    def get_node_imp(self, g):
        node_att = self.get_node_att(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, batch=g.batch, imp_edge_index=g.imp_edge_index, graph_central_node=g.graph_central_node)
        return node_att
    
    def get_graph_emb(self, g, node_att=None):
        if node_att is None:
            node_att = self.get_node_att(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, batch=g.batch, imp_edge_index=g.imp_edge_index, graph_central_node=g.graph_central_node)
        graph_emb = self.glad.get_graph_emb(x=g.x, edge_index=g.edge_index, batch=g.batch, node_imp=node_att)
        return graph_emb
        
    def get_anomaly_score(self, g):
        node_att = self.get_node_att(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, batch=g.batch, imp_edge_index=g.imp_edge_index, graph_central_node=g.graph_central_node)
        anomaly_score = self.glad.get_anomaly_score(g, node_imp=node_att)
        return anomaly_score
    
    
    def generate_triplets(self, neg_g):
        """
        生成self.pur_num组三元组，每组三元组包含：
        - pos_pur_mask: 在标签为0的节点中随机选择，组合中心节点，并用self.pos_att_aggregation保持连通性
        - neg_pur_mask: 在标签为1的节点中随机选择，组合中心节点，并用self.pos_att_aggregation保持连通性
        - all_pur_mask: pos_pur_mask和neg_pur_mask的并集，且除中心节点外两者都至少有一个节点
        """
        node_labels = neg_g.node_labels  # 假设标签存在于neg_g.node_labels
        center_mask = (node_labels == -1)
        pos_mask = (node_labels == 0)
        neg_mask = (node_labels == 1)
        num_nodes = node_labels.shape[0]
        center_idx = torch.where(center_mask)[0]

        triplets = []
        while len(triplets) < self.pur_num:
            pos_indices = torch.where(pos_mask)[0]
            pos_pur_mask = torch.zeros(num_nodes, dtype=torch.float32, device=node_labels.device)
            if len(pos_indices) != 0:
                num_pos = torch.randint(1, len(pos_indices) + 1, ())
                pos_sel = pos_indices[torch.randperm(len(pos_indices))[:num_pos]]
                pos_pur_mask[pos_sel] = 1.
                pos_pur_mask = self.pos_att_aggregation(pos_pur_mask.float(), neg_g.imp_edge_index, neg_g.graph_central_node)

            neg_indices = torch.where(neg_mask)[0]
            neg_pur_mask = torch.zeros(num_nodes, dtype=torch.float32, device=node_labels.device)
            if len(neg_indices) != 0:
                num_neg = torch.randint(1, len(neg_indices) + 1, ())
                neg_sel = neg_indices[torch.randperm(len(neg_indices))[:num_neg]]
                neg_pur_mask[neg_sel] = 1.
                neg_pur_mask = self.pos_att_aggregation(neg_pur_mask.float(), neg_g.imp_edge_index, center_idx)

            all_pur_mask = pos_pur_mask + neg_pur_mask 
            pos_pur_mask[center_idx] = 1.
            neg_pur_mask[center_idx] = 1.
            all_pur_mask[center_idx] = 1.
            triplets.append((pos_pur_mask, neg_pur_mask, all_pur_mask))
        return triplets


class PosAttPathLayer(MessagePassing):
    def __init__(self):
        super(PosAttPathLayer, self).__init__(aggr='max')

    def forward(self, initial_weight, imp_edge_index, graph_central_node):
        updated_weights = initial_weight.reshape(-1, 1)
        imp_edge_index = imp_edge_index.flip(0)
        
        for _ in range(3):
            updated_weights = self.propagate(edge_index=imp_edge_index, x=updated_weights)
        
        updated_weights[graph_central_node] = 1
        
        return updated_weights.reshape(-1)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        new_weight = torch.where(aggr_out > x, aggr_out, x)
        return new_weight
    

class NegAttPathLayer(MessagePassing):
    def __init__(self):
        super(NegAttPathLayer, self).__init__(aggr='max')

    def forward(self, initial_weight, imp_edge_index, graph_central_node):
        updated_weights = initial_weight.reshape(-1, 1)
        
        for _ in range(3):
            updated_weights = self.propagate(edge_index=imp_edge_index, x=updated_weights)
        
        updated_weights[graph_central_node] = 1
        
        return updated_weights.reshape(-1)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        no_incoming_edges = torch.isinf(aggr_out) | (aggr_out <= 0)
        new_weight = torch.where(no_incoming_edges, x, torch.min(aggr_out, x))
        return new_weight


