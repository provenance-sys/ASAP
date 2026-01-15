from typing import Dict, List, Union
import networkx as nx
import os
import sys


__all__ = ["OPREventModel", "OPRNodeModel", "OPREdgeModel"]

class OPREventModel(dict):
    def __init__(self, u: Dict = None, v: Dict = None, e: Dict = None):
        '''OPR Event Model, Dict type of the event.

        Including the attributes of the source node, target node, and edge.

        NOTE Ensure the one of the nodes is the central node type,
        for define the subject and object.

        Parameters
        ------
        u: Dict
            The source node of the event.
        v: Dict
            The target node of the event.
        e: Dict
            The edge of the event.
        '''
        super().__init__()
        if u is None: u = {}
        if v is None: v = {}
        if e is None: e = {}
        self.u = OPRNodeModel(**u)
        self.v = OPRNodeModel(**v)
        self.e = OPREdgeModel(**e)
        self.update({
            'u': self.u,
            'v': self.v,
            'e': self.e
        })

    # ============UPDATE=========== #
    def update_from_graph_edge(self, g: Union[nx.Graph, nx.MultiGraph], u, v, k = None):
        '''Update the event model from the networkx graph.

        Parameters
        ------
        g: nx.Graph, nx.MultiGraph
            The graph to update the event model.
        u:
            The source node of the edge.
        v:
            The target node of the edge.
        k:
            The key of the edge. If the graph is not a multigraph, k should be None.
        '''
        u_node = g.nodes[u]
        v_node = g.nodes[v]
        if g.is_multigraph():
            edge = g.edges[u, v, k]
        else:
            edge = g.edges[u, v]
        self.update_e(**edge)
        self.update_u(**u_node)
        self.update_v(**v_node)

    def update_from_loprem(self, loprem: List):
        '''Update the event model from the list form of oprem.'''
        u_vid, u_nid, u_type, u_content, u_flag, \
        v_vid, v_nid, v_type, v_content, v_flag, \
        edge_type, edge_ts, edge_te = loprem
        u_flag = int(u_flag)
        v_flag = int(v_flag)
        edge_ts = int(edge_ts)
        edge_te = int(edge_te)
        self.update_u(vid=u_vid, nid=u_nid, type=u_type, content=u_content, flag=u_flag)
        self.update_v(vid=v_vid, nid=v_nid, type=v_type, content=v_content, flag=v_flag)
        self.update_e(type=edge_type, ts=edge_ts, te=edge_te)

    def update_u(self, **kwargs):
        '''If the key is in the 'u', update the value of the key.'''
        self.u.update_node(**kwargs)

    def update_v(self, **kwargs):
        '''If the key is in the 'v', update the value of the key.'''
        self.v.update_node(**kwargs)
    
    def update_e(self, **kwargs):
        '''If the key is in the 'e', update the value of the key.'''
        self.e.update_edge(**kwargs)
    
    # ============QUERY=========== #
    def get_u(self, key: str):
        return self['u'].get(key)
    
    def get_v(self, key: str):
        return self['v'].get(key)
    
    def get_e(self, key: str):
        return self['e'].get(key)
    
    def get_(self, key1: str, key2: str):
        try:
            return self[key1][key2]
        except KeyError:
            return None

class OPRNodeModel(dict):
    def __init__(self, **kwargs):
        '''OPR Node Model, Dict type of the node.

        Including the attributes of the node.
        '''
        super().__init__()
        init_dict = {
            # 'ts': None,  # int
            # 'te': None,  # int
            'vid': None,  # int
            'nid': None,  # str
            'type': None,  # str
            'content': None,  # str
            'flag': None  # int
        }
        self.update(init_dict)
        self.update_node(**kwargs)
    
    def update_node(self, **kwargs):
        '''If the key is in the node, update the value of the key.'''
        for k, v in kwargs.items():
            if k in self:
                self[k] = v

class OPREdgeModel(dict):
    def __init__(self, **kwargs):
        '''OPR Edge Model, Dict type of the edge.

        Including the attributes of the edge.
        '''
        super().__init__()

        init_dict = {
            'type': None,  # str
            'ts': None,  # int
            'te': None  # int
        }
        self.update(init_dict)
        self.update_edge(**kwargs)

    def update_edge(self, **kwargs):
        '''If the key is in the edge, update the value of the key.'''
        for k, v in kwargs.items():
            if k in self:
                self[k] = v

