import os
import sys
from abc import ABC, abstractmethod
import networkx as nx
import torch
from tqdm import tqdm

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../.."))
sys.path.insert(0, project_root)
from src.utils import config
from src.utils.opreventmodel import OPREventModel, OPRNodeModel, OPREdgeModel
from src.utils.provgraph import ProvGraph


__all__ = ['OpremSaveStrategy']

class SaveStrategy(ABC):
    @abstractmethod
    def save_OPREM(self, oprem: OPREventModel):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class OpremSaveStrategy(SaveStrategy):
    '''Save the OPREventModel data to oprem format'''
    def __init__(
            self, reset: bool = True,
            oprem_path: str = ""
        ):
        self.oprem_path: str = oprem_path
        dir = os.path.dirname(self.oprem_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        if reset:
            self.f = open(self.oprem_path, 'w')
        else:
            self.f = open(self.oprem_path, 'a')
    
    def save_OPREM(self, oprem: OPREventModel):
        u = oprem['u']
        v = oprem['v']
        e = oprem['e']
        u_string = '\t'.join(map(str, u.values()))
        v_string = '\t'.join(map(str, v.values()))
        e_string = '\t'.join(map(str, e.values()))
        self.f.write(u_string + '\t' + v_string + '\t' + e_string + '\t' + '\n')
    
    def close(self):
        self.f.close()
    
    def reset(self):
        self.f = open(self.oprem_path, 'w')
        self.f.close()
    
    def save_provgraph(self, pgraph: ProvGraph):
        '''Save all ProvGraph to Oprem Format'''

        edges = list(pgraph.edges(keys=True, data=True))
        edges.sort(key=lambda x: x[3].get('ts', float('inf')))
        for u, v, key, data in edges:
        # for u, v, key, data in pgraph.edges(keys=True, data=True):
            u_attrs = pgraph.nodes[u]
            v_attrs = pgraph.nodes[v]
            loprem = [u_attrs.get('vid', None), u_attrs.get('nid', None), u_attrs.get('type', None), \
                    u_attrs.get('content', None), u_attrs.get('flag', None), \
                    v_attrs.get('vid', None), v_attrs.get('nid', None), v_attrs.get('type', None), \
                    v_attrs.get('content', None), v_attrs.get('flag', None), \
                    data.get('type', None), data.get('ts', None), data.get('te', None)]

            oprem = OPREventModel()
            oprem.update_from_loprem(loprem)
            self.save_OPREM(oprem)
