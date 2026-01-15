import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch.nn import Sequential, Linear, ReLU, Linear, LayerNorm, Identity
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool, Set2Set
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GIN
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_max, scatter
import torch.multiprocessing as mp
import random
import copy
import os
import sys

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../.."))
sys.path.insert(0, project_root)
from src.models.ginconv import GINConv
from torch_geometric.nn import GINEConv, HypergraphConv, TransformerConv #, GINConv


class GLAD(nn.Module):
    def __init__(self, 
                input_node_dim, 
                input_edge_dim, 

                encoder_hidden_dim,
                encoder_num_hidden_layers,
                ):
        super(GLAD, self).__init__()

        self.encoder = ProvGNN(
            input_node_dim=input_node_dim,
            hidden_dim=encoder_hidden_dim,
            num_hidden_layers=encoder_num_hidden_layers
        )

        self.encoder_hyper = ProvHyperGNN(
            input_edge_dim=input_edge_dim, 
            hidden_dim=encoder_hidden_dim, 
            num_hidden_layers=encoder_num_hidden_layers
        )

        embedding_dim = encoder_hidden_dim * encoder_num_hidden_layers
        self.proj_head = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU(inplace=True),
                                        nn.Linear(embedding_dim, embedding_dim))
        self.proj_head_hyper = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU(inplace=True),
                                        nn.Linear(embedding_dim, embedding_dim))
        self.init_emb()
        
    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, g, node_imp=None):
        batch_graph_embedding = self.encoder(x=g.x, edge_index=g.edge_index, batch=g.batch, node_imp=node_imp)
        batch_hypergraph_embedding = self.encoder_hyper(x=g.x, edge_attr=g.edge_attr, edge_index=g.edge_index, batch=g.batch, node_imp=node_imp)

        batch_graph_embedding = self.proj_head(batch_graph_embedding)
        batch_hypergraph_embedding = self.proj_head_hyper(batch_hypergraph_embedding)

        loss_vec = self.loss_nce(batch_graph_embedding, batch_hypergraph_embedding)  
        return loss_vec * (self.get_anomaly_score(g, node_imp).detach() ** 2) 
    
    def get_anomaly_score(self, g, node_imp=None):
        batch_graph_embedding = self.encoder(x=g.x, edge_index=g.edge_index, batch=g.batch, node_imp=node_imp)
        batch_hypergraph_embedding = self.encoder_hyper(x=g.x, edge_attr=g.edge_attr, edge_index=g.edge_index, batch=g.batch, node_imp=node_imp)
        
        batch_graph_embedding = self.proj_head(batch_graph_embedding)
        batch_hypergraph_embedding = self.proj_head_hyper(batch_hypergraph_embedding)

        batch_size, _ = batch_graph_embedding.size()

        x1 = batch_graph_embedding
        x2 = batch_hypergraph_embedding

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs).clamp(min=1e-10)  # 余弦相似度矩阵
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        return (1 - pos_sim) / 2  # 0-1

    def loss_nce(self, x1, x2, temperature=0.2):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs).clamp(min=1e-10)  # 余弦相似度矩阵
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-10).clamp(min=1e-10)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-10).clamp(min=1e-10)

        loss_0 = - torch.log(loss_0 + 1e-10)
        loss_1 = - torch.log(loss_1 + 1e-10)
        loss = (loss_0 + loss_1) / 2.0
        return loss
    
    def get_graph_emb(self, x, edge_index, batch, node_imp=None):
        batch_graph_embedding = self.encoder(x=x, edge_index=edge_index, batch=batch, node_imp=node_imp)
        batch_graph_embedding = self.proj_head(batch_graph_embedding)
        return batch_graph_embedding
        

class ProvGNN(torch.nn.Module):
    def __init__(self, input_node_dim, hidden_dim, num_hidden_layers):
        super(ProvGNN, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(p=0.5)

        for i in range(self.num_hidden_layers):
            if i:
                nn = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            else:
                nn = Sequential(Linear(input_node_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            conv = GINConv(nn)
            self.convs.append(conv)

        self.pool = global_add_pool
    
    def forward(self, x, edge_index, batch, node_imp=None):
        if node_imp is None:
            node_imp = torch.ones(x.size(0), device=x.device)
        x = x * node_imp.reshape(-1, 1)

        xs = []
        for i in range(self.num_hidden_layers):
            x = F.relu(self.convs[i](
                x=x,
                edge_index=edge_index))
            x = self.dropout(x)
            x = x * node_imp.reshape(-1, 1)
            xs.append(x)
        gh = torch.cat([self.pool(x, batch) for x in xs], 1)
        return gh
    
    def get_node_emb(self, x, edge_index, node_imp=None):
        if node_imp is None:
            node_imp = torch.ones(x.size(0), device=x.device)
        x = x * node_imp.reshape(-1, 1)

        xs = []
        for i in range(self.num_hidden_layers):
            x = F.relu(self.convs[i](
                x=x,
                edge_index=edge_index))
            x = x * node_imp.reshape(-1, 1)
            xs.append(x)
        return torch.cat([x for x in xs], 1)
    
    def get_node_last_emb(self, x, edge_index, node_imp=None):
        if node_imp is None:
            node_imp = torch.ones(x.size(0), device=x.device)
        x = x * node_imp.reshape(-1, 1)

        for i in range(self.num_hidden_layers):
            x = F.relu(self.convs[i](
                x=x,
                edge_index=edge_index))
            x = x * node_imp.reshape(-1, 1)
        return x
    

class ProvHyperGNN(torch.nn.Module):
    def __init__(self, input_edge_dim, hidden_dim, num_hidden_layers):
        super(ProvHyperGNN, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(p=0.5)

        for i in range(self.num_hidden_layers):
            if i == 0:
                conv = HypergraphConv(in_channels=input_edge_dim, out_channels=hidden_dim)
            else:
                conv = HypergraphConv(in_channels=hidden_dim, out_channels=hidden_dim)
            self.convs.append(conv)
        
        self.pool = global_add_pool

    def forward(self, x, edge_attr, edge_index, batch, node_imp=None):
        if node_imp is None:
            node_imp = torch.ones(x.size(0), device=x.device)
        edge_imp = torch.min(node_imp[edge_index[0]], node_imp[edge_index[1]])

        hyperedge_index, edge_batch, hyper_node_imp = self.get_DHT(edge_index, batch, node_imp, edge_imp)

        edge_attr = edge_attr * hyper_node_imp.reshape(-1, 1)
        xs = []
        for i in range(self.num_hidden_layers):
            edge_attr = F.relu(self.convs[i](
                x=edge_attr,
                hyperedge_index=hyperedge_index))
            edge_attr = self.dropout(edge_attr)
            edge_attr = edge_attr * hyper_node_imp.reshape(-1, 1)
            xs.append(edge_attr)
        gh = torch.cat([self.pool(x, edge_batch) for x in xs], 1)
        return gh
    
    @staticmethod
    def get_DHT(edge_index, batch, node_imp, edge_imp, add_loops=True):
        num_edge = edge_index.size(1)
        device = edge_index.device

        edge_to_node_index = torch.arange(0, num_edge, 1, device=device).repeat_interleave(2).view(1, -1)
        hyperedge_index = edge_index.T.reshape(1, -1)  # [e1_s, e1_e, e2_s, e2_e, ...]
        hyperedge_index = torch.cat([edge_to_node_index, hyperedge_index], dim=0).long()

        hyper_node_imp = edge_imp

        edge_batch = hyperedge_index[1, :].reshape(-1, 2)[:, 0]
        edge_batch = torch.index_select(batch, 0, edge_batch)

        # Add self-loops to each node in the dual hypergraph
        if add_loops:
            bincount = hyperedge_index[1].bincount()
            mask = bincount[hyperedge_index[1]] != 1
            max_edge = hyperedge_index[1].max()
            loops = torch.cat([
                torch.arange(0, num_edge, 1, device=device).view(1, -1),
                torch.arange(max_edge + 1, max_edge + num_edge + 1, 1, device=device).view(1, -1)
            ], dim=0)

            hyperedge_index = torch.cat([hyperedge_index[:, mask], loops], dim=1)

        return hyperedge_index, edge_batch, hyper_node_imp
    

class ExplainerGNN(torch.nn.Module):
    def __init__(self, input_node_dim, inupt_edge_dim, hidden_dim, num_hidden_layers, pur_num=0):
        super(ExplainerGNN, self).__init__()
        self.pur_num = pur_num

        self.num_hidden_layers = num_hidden_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(p=0.5)

        for i in range(self.num_hidden_layers):
            if i:
                nn = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            else:
                nn = Sequential(Linear(input_node_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            conv = GINConv(nn)
            self.convs.append(conv)
        
        if self.pur_num == 0:
            self.mlp = torch.nn.Linear(hidden_dim, 1)
        else:
            self.mlp = torch.nn.Linear(hidden_dim * 2, 1)
        self.pool = global_mean_pool

        self.imp_aggregation = MaxProductPathLayer(att_factor=1/2)

        self.init_emb()
        
    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, x, edge_index, edge_attr, batch, imp_edge_index, graph_central_node, node_imp=None):
        if node_imp is None:
            node_imp = torch.ones(x.size(0), device=x.device)
        x = x * node_imp.reshape(-1, 1)

        xs = []
        for i in range(self.num_hidden_layers):
            x = F.relu(self.convs[i](
                x=x,
                edge_index=edge_index, ))
            x = self.dropout(x)
            x = x * node_imp.reshape(-1, 1)
            xs.append(x)
        node_prob = 0
        for x in xs:
            node_prob += x
        
        if self.pur_num > 0:
            graph_embed = self.pool(node_prob, batch)
            graph_embed_expanded = graph_embed[batch]
            node_prob = torch.cat([node_prob, graph_embed_expanded], dim=-1)
        
        node_prob = self.mlp(node_prob)
        node_prob = torch.clamp(node_prob, min=-10.0, max=10.0)

        node_prob = self.sampling(node_prob)
        return node_prob
    
    def sampling(self, att_log_logit):
        if self.training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gated_input = torch.sigmoid((att_log_logit + random_noise) / 1)
        else:
            gated_input = torch.sigmoid(att_log_logit)
        return gated_input


class MaxProductPathLayer(MessagePassing):
    def __init__(self, att_factor = 1/2):
        super(MaxProductPathLayer, self).__init__(aggr='max')
        self.att_factor = att_factor  # attenuation factor

    def forward(self, initial_weight, imp_edge_index, graph_central_node):
        updated_weights = torch.full_like(initial_weight, float(0))
        updated_weights[graph_central_node] = 1
        self.initial_weight = torch.pow(torch.clamp(initial_weight, min=1e-10), self.att_factor)
        for _ in range(3):
            updated_weights = self.propagate(edge_index=imp_edge_index, x=updated_weights)
        
        return updated_weights

    def message(self, x_j):
        return x_j
    
    def update(self, aggr_out, x):
        new_weight = aggr_out * self.initial_weight
        return torch.where(new_weight > x, new_weight, x)
