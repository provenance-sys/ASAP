import networkx as nx
from typing import Dict, List, Optional
import copy
import os
import sys
import random
import hashlib
import xxhash
from bisect import bisect_left, bisect_right
from collections import deque
from itertools import chain

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../.."))
sys.path.insert(0, project_root)
from src.utils import config
from src.utils.opreventmodel import OPREventModel, OPRNodeModel, OPREdgeModel


__all__ = ['ProvGraph', 'RawProvGraph', 'RedProvGraph', 'DPGraph']

class ProvGraph(nx.MultiDiGraph):
    def __init__(self, central_node_type: List = config.CENTRAL_NODE_TYPE):
        """
        Provenance Graph.

        key is default allocated by networkx.
        """
        super().__init__()
        self.central_node_type = central_node_type
        self.central_nodes = set()  # current time window central nodes id
        self.non_central_nodes = set()  # current time window non central nodes id
    
    def get_central_nodes(self):
        return list(self.central_nodes)
    
    def get_non_central_nodes(self):
        return list(self.non_central_nodes)
    
    def get_all_nodes(self):
        return list(chain(self.central_nodes, self.non_central_nodes))

    def clear_central_nodes(self):
        self.central_nodes.clear()

    def clear_non_central_nodes(self):
        self.non_central_nodes.clear()

    def is_in_current_windows(self, node_id):
        return node_id in self.central_nodes or node_id in self.non_central_nodes
    
    def get_subgraph(self, node_id, num_hop):
        """
        Gets the subgraph within the specified number of hops of a node, regardless of direction.

        Parameters
        ------
        node_id: nid or vid
        num_hop: the number of hops from center node
        """
        subgraph = nx.ego_graph(self.to_undirected(as_view=True), node_id, radius=num_hop)
        return subgraph
    
    def find_reachable_subgraph_sel_node(self, start_node, hops=3, degree_limit=100):  # _modify
        forward_edges = set()
        backward_edges = set()

        forward_nodes = set()
        backward_nodes = set()
        forward_nodes.add(start_node)
        backward_nodes.add(start_node)

        imp_edges = set()

        def dfs_forward(node, depth, last_ts):
            if depth > hops:
                return
            if len(list(self.successors(node))) > degree_limit:
                return
            
            next_search_neighbors = set()
            next_search_neighbors_content_type = set() 
            for neighbor in list(self.successors(node)):
                if neighbor not in forward_nodes:
                    content_type = self.nodes[neighbor].get('type', 'None') + self.nodes[neighbor].get('content', None)
                    if content_type not in next_search_neighbors_content_type:
                        next_search_neighbors_content_type.add(content_type)
                    else:
                        continue
                    
                    min_ts = None
                    for key, attrs in self.get_edge_data(node, neighbor).items():
                        if attrs['ts'] < last_ts:
                            continue
                        if min_ts is None or min_ts > attrs['ts']:
                            min_ts = attrs['ts']
                        forward_edges.add((node, neighbor, key))

                    if min_ts is not None:
                        if neighbor not in forward_nodes:
                            next_search_neighbors.add((neighbor, min_ts))
                            forward_nodes.add(neighbor)
                        if (neighbor, node) not in imp_edges:
                            imp_edges.add((node, neighbor))
            if depth < hops:
                for neighbor, _ts in next_search_neighbors:
                    dfs_forward(neighbor, depth + 1, _ts)
        
        neig_num = len(list(self.successors(start_node)))
        next_search_neighbors = set()
        for neighbor in list(self.successors(start_node)):
            forward_nodes.add(neighbor)
            if (neighbor, start_node) not in imp_edges:
                imp_edges.add((start_node, neighbor))

            min_ts = None
            for key, attrs in self.get_edge_data(start_node, neighbor).items():
                if min_ts is None or min_ts > attrs['ts']:
                    min_ts = attrs['ts']
                forward_edges.add((start_node, neighbor, key))
            next_search_neighbors.add((neighbor, min_ts))
        if neig_num < degree_limit:
            for neighbor, _ts in next_search_neighbors:
                dfs_forward(neighbor, 2, _ts)
        
        def dfs_backward(node, depth, last_ts):
            if depth > hops:
                return
            if len(list(self.predecessors(node))) > degree_limit:
                return

            next_search_neighbors = set()
            next_search_neighbors_content_type = set()
            for neighbor in list(self.predecessors(node)):
                if neighbor not in backward_nodes:

                    content_type = self.nodes[neighbor].get('type', 'None') + self.nodes[neighbor].get('content', None)
                    if content_type not in next_search_neighbors_content_type:
                        next_search_neighbors_content_type.add(content_type)
                    else:
                        continue

                    max_ts = None
                    for key, attrs in self.get_edge_data(neighbor, node).items():
                        if attrs['ts'] > last_ts:
                            continue
                        if max_ts is None or max_ts < attrs['ts']:
                            max_ts = attrs['ts']
                        backward_edges.add((neighbor, node, key))

                    if max_ts is not None:
                        if neighbor not in backward_nodes:
                            next_search_neighbors.add((neighbor, max_ts))
                            backward_nodes.add(neighbor)
                        if (neighbor, node) not in imp_edges:
                            imp_edges.add((node, neighbor))
            
            if depth < hops:
                for neighbor, _ts in next_search_neighbors:
                    dfs_backward(neighbor, depth + 1, _ts)
            
        neig_num = len(list(self.predecessors(start_node)))
        next_search_neighbors = set()
        for neighbor in list(self.predecessors(start_node)):
            backward_nodes.add(neighbor)
            if (neighbor, start_node) not in imp_edges:
                imp_edges.add((start_node, neighbor))
            
            max_ts = None
            for key, attrs in self.get_edge_data(neighbor, start_node).items():
                if max_ts is None or max_ts < attrs['ts']:
                    max_ts = attrs['ts']
                backward_edges.add((neighbor, start_node, key))
            next_search_neighbors.add((neighbor, max_ts))
        if neig_num < degree_limit:
            for neighbor, _ts in next_search_neighbors:
                dfs_backward(neighbor, 2, _ts)

        subgraph = self.subgraph(forward_nodes | backward_nodes).copy()

        subgraph.graph['imp_edges'] = {edge for edge in imp_edges if edge[0] != edge[1]}
        return copy.deepcopy(subgraph)


class RawProvGraph(ProvGraph):
    def __init__(self, central_node_type: List = config.CENTRAL_NODE_TYPE):
        """
        Original Provenance Graph.

        nid is node name; key is default allocated by networkx.
        """
        super().__init__(central_node_type=central_node_type)
        self.all_nodes = set()
    
    def clear_processed_nodes(self):
        self.all_nodes.clear()
        self.clear()
    
    def add_new_node(self, oprnm: OPRNodeModel):
        '''Add a new node to the graph.
        
        Parameters
        ------
        oprnm: OPRNodeModel
            The attributes of the node.
            Note, it should include the 'nid', 'type'
        '''
        nid = oprnm['nid']
        self.add_node(nid, **oprnm)
    
    def add_new_edge(self, u: str, v: str, oprem: OPREdgeModel):
        '''Add a new edge to the graph.'''
        self.add_edge(u, v, **oprem)
    
    def merge_node(self, nid: str, oprnm: OPRNodeModel):
        '''Merge the node to the existing node.'''   
        ncontent = oprnm['content']
        if 'unknown' not in ncontent:
            self.nodes[nid]['content'] = ncontent
    
    def update_node(self, oprnm: OPRNodeModel):
        '''Updata the node to the graph and central_nodes'''
        nid = oprnm['nid']
        if self.has_node(nid):
            self.merge_node(nid, oprnm)
        else:
            self.add_new_node(oprnm)

        ntype = oprnm['type']
        if ntype in self.central_node_type:
            self.central_nodes.add(nid)
    
    def add_from_OPREM(self, oprem: OPREventModel):
        '''Add event from the OPREventModel to the Graph.'''
        u = oprem['u']
        v = oprem['v']
        e = oprem['e']
        u_id, v_id = u['nid'], v['nid']
        self.update_node(u)
        self.update_node(v)
        self.add_new_edge(u_id, v_id, e)

        self.all_nodes.add(u_id)
        self.all_nodes.add(v_id)


class RedProvGraph(ProvGraph):
    def __init__(self, central_node_type: List = config.CENTRAL_NODE_TYPE):
        """
        Reduced Provenance Graph.

        vid is node name; key is default allocated by networkx.
        """
        super().__init__(central_node_type=central_node_type)
        self._vid: int = 0  # version ID, monotonous increase
        self.nid2vid = {}
        self.vid2nid = {}
    
    def add_new_node(self, oprnm: OPRNodeModel):
        '''Add a new node to the graph.
        
        Parameters
        ------
        oprnm: OPRNodeModel
            The attributes of the node.
            Note, it should include the 'vid', 'type'
        '''
        vid = oprnm['vid']
        self.add_node(vid, **oprnm)
    
    def add_new_edge(self, u: str, v: str, oprem: OPREdgeModel):
        '''Add a new edge to the graph.'''
        self.add_edge(u, v, **oprem)
    
    def merge_node(self, vid: str, oprnm: OPRNodeModel):
        '''Merge the node to the existing node.'''
        
        ncontent = oprnm['content']
        if self.nodes[vid]['content'] == 'unknown' or ncontent != 'unknown' and self.nodes[vid]['content'] != ncontent:
            self.nodes[vid]['content'] = ncontent
    
    def update_node(self, oprnm: OPRNodeModel):
        '''Updata the node to the graph and central_nodes'''

        oprnm_nid_list = copy.copy(oprnm['nid']).split(';')
        oprnm_content_list = copy.copy(oprnm['content']).split(';')

        if oprnm_nid_list[0] in self.nid2vid:
            if self.vid2nid[self.nid2vid[oprnm_nid_list[0]]] == oprnm['nid']:
                _vid = self.nid2vid[oprnm_nid_list[0]]
                return [_vid]
        
        new_node_list = []

        nonempty_nid_dict = {}  # vid: [nid1, nid2, ...]
        empty_nid_list = []
        for _uid in oprnm_nid_list:
            if _uid in self.nid2vid:
                if self.nid2vid[_uid] in nonempty_nid_dict:
                    nonempty_nid_dict[self.nid2vid[_uid]].append(_uid)
                else:
                    nonempty_nid_dict[self.nid2vid[_uid]] = [_uid]
            else:
                empty_nid_list.append(_uid)
        
        for _vid in nonempty_nid_dict:
            if len(self.nodes[_vid]['nid'].split(';')) == len(nonempty_nid_dict[_vid]):
                new_node_list.append(_vid)
            else:
                new_node_vid = self._vid
                self._vid += 1
                if oprnm['type'] in self.central_node_type:
                    self.central_nodes.add(new_node_vid)

                # copy node
                self.add_node(new_node_vid, **copy.deepcopy(self.nodes[_vid]))
                for _, target, key, data in self.out_edges(_vid, keys=True, data=True):
                    if _vid != target:
                        self.add_edge(new_node_vid, target, key=key, **copy.deepcopy(data))
                    else:
                        self.add_edge(new_node_vid, new_node_vid, key=key, **copy.deepcopy(data))
                for source, _, key, data in self.in_edges(_vid, keys=True, data=True):
                    if _vid != source:
                        self.add_edge(source, new_node_vid, key=key, **copy.deepcopy(data))
                
                # Modify the original node nid
                _vid_nid_list = self.nodes[_vid]['nid'].split(';')
                indices = [_vid_nid_list.index(x) for x in nonempty_nid_dict[_vid]]
                for i in sorted(indices, reverse=True):
                    del _vid_nid_list[i]
                self.nodes[_vid]['nid'] = ';'.join(list(_vid_nid_list))

                # Modify the original node content
                _vid_content_list = self.nodes[_vid]['content'].split(';')
                if len(_vid_content_list) > 1:
                    for i in sorted(indices, reverse=True):
                        del _vid_content_list[i]
                    self.nodes[_vid]['content'] = ';'.join(list(_vid_content_list))
                
                # if the original node has no content, remove it; or Modify vid2nid
                if len(_vid_nid_list) == 0 or len(self.nodes[_vid]['nid']) == 0:
                    self.remove_node(_vid)
                    del self.vid2nid[_vid]
                    if oprnm['type'] in self.central_node_type:
                        self.central_nodes.remove(_vid)
                else:
                    self.vid2nid[_vid] = self.nodes[_vid]['nid'] 
                
                # Modify the new node nid
                self.nodes[new_node_vid]['nid'] = ';'.join(nonempty_nid_dict[_vid])

                # Modify the new node content
                if len(oprnm_content_list) > 1:
                    indices = [oprnm_nid_list.index(x) for x in nonempty_nid_dict[_vid]]
                    self.nodes[new_node_vid]['content'] = ';'.join(list([oprnm_content_list[i] for i in indices]))
                else:
                    self.nodes[new_node_vid]['content'] = oprnm['content']
                
                # Modify the new node vid2nid and nid2vid
                self.vid2nid[new_node_vid] = self.nodes[new_node_vid]['nid']
                for _nid in nonempty_nid_dict[_vid]:
                    self.nid2vid[_nid] = new_node_vid
                
                new_node_list.append(new_node_vid)

        if len(empty_nid_list) > 0:
            indices = [oprnm_nid_list.index(x) for x in empty_nid_list]
            if len(oprnm_content_list) > 1:
                oprnm['content'] = ';'.join(list([oprnm_content_list[i] if i < len(oprnm_content_list) else 'none' for i in indices]))
            oprnm['nid'] = ';'.join(list(empty_nid_list))
            oprnm['vid'] = self._vid
            self._vid += 1
            if oprnm['type'] in self.central_node_type:
                self.central_nodes.add(oprnm['vid'])
            self.add_new_node(oprnm)

            self.vid2nid[oprnm['vid']] = oprnm['nid']
            for _nid in empty_nid_list:
                self.nid2vid[_nid] = oprnm['vid']

            new_node_list.append(oprnm['vid'])
        return new_node_list
    
    def add_from_OPREM(self, oprem: OPREventModel):
        '''Add event from the OPREventModel to the CacheGraph.'''
        u = oprem['u']
        v = oprem['v']
        e = oprem['e']

        u_node_list = self.update_node(u)
        v_node_list = self.update_node(v)

        for _u in u_node_list:
            for _v in v_node_list:
                self.add_new_edge(_u, _v, e)
    

class DPGraph(ProvGraph):
    def __init__(self, central_node_type: List = config.CENTRAL_NODE_TYPE,
                is_merge_entity: bool = True,):
        """
        Convert Raw Provenance Graph (nid) to Reduced Dependence-Preserving Graph (vid).

        vid is node name; key is default allocated by networkx.
        """
        super().__init__(central_node_type=central_node_type)
        self._vid: int = 0  # version ID, monotonous increase
        self.__nid2lvid: Dict[str, int] = {}

        self.current_latest_time = None
        self.previous_window_time = None

        self.is_merge_entity = is_merge_entity
        if is_merge_entity:
            self.__central_nodes_hash: Dict[int, int] = {}  # vid, hash
            self.__non_central_nodes_hash: Dict[int, int] = {}  # vid, hash
    
    # ============CLEARING WINDOWS NODES=========== #
    def clear_processed_nodes(self):
        self.clear_central_nodes()
        self.clear_non_central_nodes()
        self.clear()
        self.previous_window_time = self.current_latest_time
        self.__nid2lvid: Dict[str, int] = {}
        # self.current_latest_time = None
    
    # ============__nid2lvid VID AND NID=========== #
    def vid2nid(self, vid: int) -> str:
        '''Get the node ID from the version ID.'''
        if self.has_node(vid):
            return self.nodes[vid]['nid']
        else:
            return None

    def nid2vid(self, nid: str) -> Optional[int]:
        '''
        if nid in the graph, return the version ID of the node.
        if nid not in the graph, return None.
        '''
        return self.__nid2lvid.get(nid, None)
    
    def update_vid(self, nid: str, vid: int):
        '''Add or Update the version ID of the node.'''
        self.__nid2lvid[nid] = vid
    
    # ============ADD NODE=========== #
    def add_new_node(self, oprnm: OPRNodeModel) -> int:
        '''Add new node to DPGraph, return the version ID of the node.'''
        # mutable object need to be copied
        oprnm = copy.copy(oprnm)
        nid = oprnm['nid']
        self._vid += 1  # update the version count
        self.update_vid(nid, self._vid)

        oprnm['vid'] = self._vid
        self.add_node(self._vid, **oprnm)
        return self._vid
    
    def add_new_version_node(self) -> int:
        self._vid += 1  # update the version count
        self.add_node(self._vid)
        return self._vid
    
    def update_node(self, oprnm: OPRNodeModel) -> int:
        vid = self.nid2vid(oprnm['nid'])

        if vid is not None and self.is_in_current_windows(vid):
            ncontent = oprnm['content']
            if self.nodes[vid]['content'] == 'unknown' or ncontent != 'unknown' and self.nodes[vid]['content'] != ncontent:
                self.nodes[vid]['content'] = ncontent
            return vid
        else:
            vid = self.add_new_node(oprnm)
            return vid
        
    
    # ============ADD EDGE=========== #
    def add_new_edge(self, u_vid: int, v_vid: int, oprem: OPREdgeModel):
        '''Add a new edge to the graph.'''
        self.add_edge(u_vid, v_vid, **oprem)
    
    # ============ADD EVENT=========== #
    def add_from_OPREM(self, oprem: OPREventModel):
        '''Add event from the OPREventModel to the CacheGraph.'''
        u = oprem['u']
        v = oprem['v']
        e = oprem['e']

        # add node
        u_vid = self.update_node(u)
        if u['type'] in self.central_node_type:
            self.central_nodes.add(u_vid)
        else:
            self.non_central_nodes.add(u_vid)

        v_vid = self.update_node(v)
        if v['type'] in self.central_node_type:
            self.central_nodes.add(v_vid)
        else:
            self.non_central_nodes.add(v_vid)

        # add edge
        self.add_new_edge(u_vid, v_vid, e)

        # update the latest time
        if self.current_latest_time is None or self.current_latest_time < e['ts']:
            self.current_latest_time = e['ts']

        # update the node hash
        if u['type'] in self.central_node_type:
            self.update_central_node_hash(u_vid, e.get('type', 'None'), v_vid, "out")
        else:
            self.update_non_central_node_hash(u_vid, e.get('type', 'None'), v_vid, "out")
        if v['type'] in self.central_node_type:
            self.update_central_node_hash(v_vid, e.get('type', 'None'), u_vid, "in")
        else:
            self.update_non_central_node_hash(v_vid, e.get('type', 'None'), u_vid, "in")
    
    def update_central_node_hash(self, vid, edge_type, neig_vid, edge_dir, P=31, M=2**64):
        '''Update the hash of the process node when adding events'''
        if vid not in self.__central_nodes_hash:
            node_content = self.nodes[vid].get('content', 'None')
            self.__central_nodes_hash[vid] = self.cal_string_hash(f"{node_content}")
        
        if vid == neig_vid:
            edge_hash = self.cal_string_hash(f"{edge_type}, selfloop")
        else:
            edge_hash = self.cal_string_hash(f"{edge_type},{edge_dir},{neig_vid}")
        self.__central_nodes_hash[vid] = (self.__central_nodes_hash[vid] * P + edge_hash) % M
    
    def update_non_central_node_hash(self, vid, edge_type, neig_vid, edge_dir, P=31, M=2**64):
        '''Update the hash of the non_central node when adding events'''
        if vid not in self.__non_central_nodes_hash:
            node_content = self.nodes[vid].get('type', 'None')  
            self.__non_central_nodes_hash[vid] = self.cal_string_hash(f"{node_content}")  # Merge without considering the node content
        
        if vid == neig_vid:
            edge_hash = self.cal_string_hash(f"{edge_type}, selfloop")
        else:
            edge_hash = self.cal_string_hash(f"{edge_type},{edge_dir},{neig_vid}")
        self.__non_central_nodes_hash[vid] = (self.__non_central_nodes_hash[vid] * P + edge_hash) % M

    def cal_string_hash(self, s: str) -> int:
        '''Calculates the MD5/SHA256/XXH64 hash of a string'''
        # return int(hashlib.md5(s.encode()).hexdigest(), 16)
        # return int(hashlib.sha256(s.encode()).hexdigest(), 16)
        return xxhash.xxh64(s).intdigest()

    # ============PROCESS REDUCTION=========== #
    def central_node_reduction(self):
        '''Merge central nodes with identical attributes and neighborhood'''
        hash_to_central_node_list = {}  # {hash1: [vid1, vid2], hash2: [vid3, vid4], ...}
        for _vid in self.central_nodes:
            node_hash = self.__central_nodes_hash[_vid]
            node_hash = (node_hash * 31 + self.cal_string_hash(str(self.in_degree(_vid)))) % (2**64)
            node_hash = (node_hash * 31 + self.cal_string_hash(str(self.out_degree(_vid)))) % (2**64)
            if node_hash in hash_to_central_node_list:
                hash_to_central_node_list[node_hash].append(_vid)
            else:
                hash_to_central_node_list[node_hash] = [_vid]
        
        for _vids in hash_to_central_node_list.values():
            if len(_vids) < 2: 
                continue
            reduced_central_node, sorted_vids = self.cal_reduced_central_node(_vids)

            current_vid = sorted_vids[0]
            for _vid in sorted_vids:
                if _vid in reduced_central_node:
                    self.__nid2lvid[self.nodes[_vid]['nid']] = current_vid
                    self.nodes[current_vid]['nid'] = self.nodes[current_vid]['nid'] + ';' + self.nodes[_vid]['nid']
                    
                    self.remove_node(_vid)
                    self.central_nodes.remove(_vid)
                else:
                    current_vid = _vid

    def cal_reduced_central_node(self, vids: List) -> List:
        """
        Returns the central nodes to be merged

        vids is a set of central nodes with identical attributes and neighborhood
        """
        vids_out_edge = {}  # {vid1: [[neig1, key1, ts1], [neig2, key2, ts2]], vid2: [], ...}
        vids_in_edge = {}
        for _vid in vids:
            vids_out_edge[_vid] = []
            for u, v, key, data in self.out_edges(_vid, keys=True, data=True):
                if u != v and v not in vids:
                    vids_out_edge[_vid].append([v, key, data.get('ts')])
            vids_out_edge[_vid] = sorted(vids_out_edge[_vid], key=lambda x: x[2])

            vids_in_edge[_vid] = []
            for u, v, key, data in self.in_edges(_vid, keys=True, data=True):
                if u != v and u not in vids:
                    vids_in_edge[_vid].append([u, key, data.get('ts')])
            vids_in_edge[_vid] = sorted(vids_in_edge[_vid], key=lambda x: x[2])
        
        in_neighbors = set(self.predecessors(vids[0])) - set(vids)
        out_neighbors = set(self.successors(vids[0])) - set(vids)

        in_neighbors_in_edge_ts = {}  # {neig1: [ts1, ts2], neig2: []..}
        out_neighbors_out_edge_ts = {}

        for _in_neig in in_neighbors:
            edges = self.in_edges(_in_neig, keys=True, data=True)
            in_neighbors_in_edge_ts[_in_neig] = []
            for u, v, key, data in edges:
                if u not in vids:
                    in_neighbors_in_edge_ts[_in_neig].append(data.get('ts'))
            in_neighbors_in_edge_ts[_in_neig] = sorted(in_neighbors_in_edge_ts[_in_neig])

        for _out_neig in out_neighbors:
            edges = self.out_edges(_out_neig, keys=True, data=True)
            out_neighbors_out_edge_ts[_out_neig] = []
            for u, v, key, data in edges:
                if v not in vids:
                    out_neighbors_out_edge_ts[_out_neig].append(data.get('ts'))
            out_neighbors_out_edge_ts[_out_neig] = sorted(out_neighbors_out_edge_ts[_out_neig])

        preserved_nodes = set()
        sorted_vids = None
        for i in range(len(vids_out_edge[vids[0]])):
            merged_nodes = []  # [[vid1, ts1], [vid2, ts2], ...]
            for vid, values in vids_out_edge.items():
                merged_nodes.append([vid, values[i][2]])
            preserved_i_nodes = self.cal_preserved_object(merged_nodes, out_neighbors_out_edge_ts[vids_out_edge[vids[0]][i][0]])
            preserved_nodes.update(preserved_i_nodes)

            if sorted_vids is None:
                merged_nodes.sort(key=lambda x: x[1])
                sorted_vids = [x[0] for x in merged_nodes]
        
        for i in range(len(vids_in_edge[vids[0]])):
            merged_nodes = []  # [[vid1, ts1], [vid2, ts2], ...]
            for vid, values in vids_in_edge.items():
                merged_nodes.append([vid, values[i][2]])
            preserved_i_nodes = self.cal_preserved_object(merged_nodes, in_neighbors_in_edge_ts[vids_in_edge[vids[0]][i][0]])
            preserved_nodes.update(preserved_i_nodes)

            if sorted_vids is None:
                merged_nodes.sort(key=lambda x: x[1])
                sorted_vids = [x[0] for x in merged_nodes]

        # solitary node
        if sorted_vids is None:
            return set(vids) - set([vids[0]]), vids

        return set(vids) - preserved_nodes, sorted_vids
    
    def cal_preserved_object(self, merged_objs, combined_ts):
        """
        Compute the objects to be saved after interleaved information flow
        """
        merged_objs.sort(key=lambda x: x[1])
        preserved_objs = {merged_objs[0][0]}
        merged_objs_time = [item[1] for item in merged_objs]
        if len(combined_ts) > 0 and combined_ts[-1] > merged_objs[0][1]:
            for _ts in combined_ts:
                if _ts < merged_objs[0][1]:
                    continue
                if _ts >= merged_objs[-1][1]:
                    break
                _index = bisect_right(merged_objs_time, _ts)
                preserved_objs.add(merged_objs[_index][0])
        return set(preserved_objs)
    
    # ============OBJECT REDUCTION=========== #
    def non_central_node_reduction(self):
        '''Merge file nodes with identical attributes and neighborhood'''
        hash_to_non_central_node_list = {}  # {hash1: [vid1, vid2], hash2: [vid3, vid4], ...}
        for _vid in self.non_central_nodes:
            node_hash = self.__non_central_nodes_hash[_vid]
            node_hash = (node_hash * 31 + self.cal_string_hash(str(self.in_degree(_vid)))) % (2**64)
            node_hash = (node_hash * 31 + self.cal_string_hash(str(self.out_degree(_vid)))) % (2**64)
            if node_hash in hash_to_non_central_node_list:
                hash_to_non_central_node_list[node_hash].append(_vid)
            else:
                hash_to_non_central_node_list[node_hash] = [_vid]
        
        for _vids in hash_to_non_central_node_list.values():
            if len(_vids) < 2: 
                continue
            reduced_non_central_node, sorted_vids = self.cal_reduced_non_central_node(_vids)
            if sorted_vids is None:
                continue

            # merge content
            current_vid = sorted_vids[0]
            for _vid in sorted_vids:
                if _vid in reduced_non_central_node:
                    self.__nid2lvid[self.nodes[_vid]['nid']] = current_vid
                    self.nodes[current_vid]['nid'] = self.nodes[current_vid]['nid'] + ';' + self.nodes[_vid]['nid']
                    self.nodes[current_vid]['content'] = self.nodes[current_vid]['content'] + ';' + self.nodes[_vid]['content']
                    
                    self.remove_node(_vid)
                    self.non_central_nodes.remove(_vid)
                else:
                    current_vid = _vid
    
    def cal_reduced_non_central_node(self, vids: List) -> List:
        """
        Returns the file nodes to be merged

        vids is a set of file nodes with identical attributes and neighborhood
        """
        vids_out_edge = {}  # {vid1: [[neig1, key1, ts1], [neig2, key2, ts2]], vid2: [], ...}
        vids_in_edge = {}

        for _vid in vids:
            vids_out_edge[_vid] = []
            for u, v, key, data in self.out_edges(_vid, keys=True, data=True):
                if u != v:
                    vids_out_edge[_vid].append([v, key, data.get('ts')])
            vids_out_edge[_vid] = sorted(vids_out_edge[_vid], key=lambda x: x[2])


            vids_in_edge[_vid] = []
            for u, v, key, data in self.in_edges(_vid, keys=True, data=True):
                if u != v:
                    vids_in_edge[_vid].append([u, key, data.get('ts')])
            vids_in_edge[_vid] = sorted(vids_in_edge[_vid], key=lambda x: x[2])
        
        in_neighbors = set(self.predecessors(vids[0])) - set(vids)
        out_neighbors = set(self.successors(vids[0])) - set(vids)

        in_neighbors_in_edge_ts = {}  # {neig1: [ts1, ts2], neig2: []..}
        out_neighbors_out_edge_ts = {}

        for _in_neig in in_neighbors:
            edges = self.in_edges(_in_neig, keys=True, data=True)
            in_neighbors_in_edge_ts[_in_neig] = []
            for u, v, key, data in edges:
                if u not in vids:
                    in_neighbors_in_edge_ts[_in_neig].append(data.get('ts'))
            in_neighbors_in_edge_ts[_in_neig] = sorted(in_neighbors_in_edge_ts[_in_neig])

        for _out_neig in out_neighbors:
            edges = self.out_edges(_out_neig, keys=True, data=True)
            out_neighbors_out_edge_ts[_out_neig] = []
            for u, v, key, data in edges:
                if v not in vids:
                    out_neighbors_out_edge_ts[_out_neig].append(data.get('ts'))
            out_neighbors_out_edge_ts[_out_neig] = sorted(out_neighbors_out_edge_ts[_out_neig])
        
        preserved_nodes = set()
        sorted_vids = None
        for i in range(len(vids_out_edge[vids[0]])):
            merged_nodes = []  # [[vid1, ts1], [vid2, ts2], ...]
            for vid, values in vids_out_edge.items():
                # if len(values) > i:  # ？？？
                merged_nodes.append([vid, values[i][2]])
            preserved_i_nodes = self.cal_preserved_object(merged_nodes, out_neighbors_out_edge_ts[vids_out_edge[vids[0]][i][0]])
            preserved_nodes.update(preserved_i_nodes)

            if sorted_vids is None:
                merged_nodes.sort(key=lambda x: x[1])
                sorted_vids = [x[0] for x in merged_nodes]
        
        for i in range(len(vids_in_edge[vids[0]])):
            merged_nodes = []  # [[vid1, ts1], [vid2, ts2], ...]
            for vid, values in vids_in_edge.items():
                merged_nodes.append([vid, values[i][2]])
            preserved_i_nodes = self.cal_preserved_object(merged_nodes, in_neighbors_in_edge_ts[vids_in_edge[vids[0]][i][0]])
            preserved_nodes.update(preserved_i_nodes)

            if sorted_vids is None:
                merged_nodes.sort(key=lambda x: x[1])
                sorted_vids = [x[0] for x in merged_nodes]

        return set(vids) - preserved_nodes, sorted_vids
    
    # ============DUPLICATIVE EVENT REDUCTION=========== #
    def duplicative_event_reduction(self):
        '''Merge duplicate events'''
        for _vid in self.central_nodes:
            # Merge duplicate events
            successors = list(self.successors(_vid))
            for neig in successors:
                self.merge_event(_vid, neig)

            predecessors = list(self.predecessors(_vid))
            for neig in predecessors:
                self.merge_event(neig, _vid)

            if self.has_edge(_vid, _vid):
                self.merge_event(_vid, _vid)
    
    def merge_event(self, u, v):
        edges = self.get_edge_data(u, v)
        if len(edges) == 1:
            return
        
        is_duplicate = False
        event_edges = {}  # {edge_type1: [key1, key2], edge_type2: [], ...}
        for key, data in edges.items():
            edge_type = data.get('type')
            if edge_type not in event_edges:
                event_edges[edge_type] = [key]
            else:
                event_edges[edge_type].append(key)
                is_duplicate = True
        if is_duplicate is False:
            return
        
        if u != v:
            combined_ts = {data for _, _, data in (self.in_edges(u, data='ts') or []) if data is not None} | \
                            {data for _, _, data in (self.out_edges(v, data='ts') or []) if data is not None}
            combined_ts = sorted(combined_ts)  # [ts1, ts2, ...]

            for keys in event_edges.values():
                if len(keys) == 1:
                    continue

                merged_edges = []  # [[key1, ts1], [key2, ts2], ...]
                for key in keys:
                    merged_edges.append([key, self.get_edge_data(u, v, key).get('ts')])
                merged_edges.sort(key=lambda x: x[1])

                preserved_edges = self.cal_preserved_object(merged_edges, combined_ts)
                for _key in set(keys) - preserved_edges:
                    self.remove_edge(u, v, key=_key)
        
        else:
            combined_ts = {data for src, tgt, data in (self.in_edges(u, data='ts') or []) if data is not None and src != tgt} | \
                            {data for src, tgt, data in (self.out_edges(v, data='ts') or []) if data is not None and src != tgt}
            combined_ts = sorted(combined_ts)

            for edge_type, keys in event_edges.items():
                if len(keys) == 1:
                    continue

                combined_ts_loop = set(combined_ts) | {data.get('ts', None) for src, tgt, data in (self.in_edges(u, data=True) or []) if data is not None and (src == tgt and data.get('type', None) != edge_type)}
                combined_ts_loop = sorted(combined_ts_loop)

                merged_edges = []  # [[key1, ts1], [key2, ts2], ...]
                for key in keys:
                    merged_edges.append([key, self.get_edge_data(u, v, key).get('ts')])
                merged_edges.sort(key=lambda x: x[1])

                preserved_edges = self.cal_preserved_object(merged_edges, combined_ts_loop)
                for _key in set(keys) - preserved_edges:
                    self.remove_edge(u, v, key=_key)
    
    # ============MULTI DUPLICATIVE EVENT REDUCTION=========== #
    def multi_duplicative_event_reduction(self):
        '''Merge duplicate read&write events'''
        for _vid in self.central_nodes:
            if self.degree(_vid) < 20:
                continue
            self.merge_multi_event(_vid)

        for _vid in self.non_central_nodes:
            if self.degree(_vid) < 20:
                continue
            self.merge_multi_event(_vid)
    
    def merge_multi_event(self, vid):
        # Process the incoming edge
        in_edges_dict = {}  # {[neig, edge_type]: [[edge_key1, ts1], [edge_key2, ts2], ...], []:[], ...}
        in_edges = self.in_edges(vid, data=True, keys=True)
        for u, v, key, data in in_edges:
            edge_type = data.get('type', 'None')
            edge_ts = data.get('ts', None)

            if (u, edge_type) not in in_edges_dict:
                in_edges_dict[(u, edge_type)] = [[key, edge_ts]]
            else:
                in_edges_dict[(u, edge_type)].append([key, edge_ts])
        
        max_in_edges = None
        if in_edges_dict:
            max_in_edges = max(in_edges_dict, key=lambda k: len(in_edges_dict[k]))
        
        if max_in_edges is None or len(in_edges_dict[max_in_edges]) < 2:
            return

        # Process the outgoing edge
        out_edges_dict = {}  # {[neig, edge_type]: [[edge_key1, ts1], [edge_key2, ts2], ...], []:[], ...}
        out_edges = self.out_edges(vid, data=True, keys=True)
        for u, v, key, data in out_edges:
            edge_type = data.get('type', 'None')
            edge_ts = data.get('ts', None)

            if (v, edge_type) not in out_edges_dict:
                out_edges_dict[(v, edge_type)] = [[key, edge_ts]]
            else:
                out_edges_dict[(v, edge_type)].append([key, edge_ts])
        
        max_out_edges = None
        if out_edges_dict:
            max_out_edges = max(out_edges_dict, key=lambda k: len(out_edges_dict[k]))

        if max_out_edges is None or len(out_edges_dict[max_out_edges]) < 2:
            return
        
        # Create vid multiple versions
        vid_multi_version = {}
        if len(in_edges_dict[max_in_edges]) >= len(out_edges_dict[max_out_edges]):
            in_edges_dict[max_in_edges] = sorted(in_edges_dict[max_in_edges], key=lambda x: x[1])
            for _edge_key, _ts in in_edges_dict[max_in_edges]:
                vid_multi_version[_ts] = []
        else:
            out_edges_dict[max_out_edges] = sorted(out_edges_dict[max_out_edges], key=lambda x: x[1])
            for _edge_key, _ts in out_edges_dict[max_out_edges]:
                vid_multi_version[_ts] = []
        
        # Adding the edges of vid node to version points
        sorted_vid_multi_version_ts = sorted(vid_multi_version.keys())
        for u, v, key, data in in_edges:
            edge_type = data.get('type', 'None')
            edge_ts = data.get('ts', None)
            bisect_index = bisect_right(sorted_vid_multi_version_ts, edge_ts)
            if bisect_index == 0:
                ts_index = 0
            else:
                ts_index = bisect_index - 1
            
            vid_multi_version[sorted_vid_multi_version_ts[ts_index]].append([u, key, edge_ts, edge_type, 'in'])
        
        for u, v, key, data in out_edges:
            edge_type = data.get('type', 'None')
            edge_ts = data.get('ts', None)
            bisect_index = bisect_right(sorted_vid_multi_version_ts, edge_ts)
            if bisect_index == 0:
                ts_index = 0
            else:
                ts_index = bisect_index - 1
            
            vid_multi_version[sorted_vid_multi_version_ts[ts_index]].append([v, key, edge_ts, edge_type, 'out'])

        # Calculate each version point hash value and creat version node
        vid_multi_version_hash_old = None
        version_node_vid_list = []
        version_node_vid2key = {}
        for version_key in sorted_vid_multi_version_ts:
            version_node_vid = self.add_new_version_node()
            vid_multi_version_hash = 1

            vid_multi_version[version_key] = sorted(vid_multi_version[version_key], key=lambda x: x[2])
            for _neig, _key, _ts, _type, _mode in vid_multi_version[version_key]:

                version_edge = OPREdgeModel(type=_type, ts=_ts, te=_ts)
                if _mode == 'in':
                    self.add_edge(_neig, version_node_vid, **version_edge)
                else:
                    self.add_edge(version_node_vid, _neig, **version_edge)

                _hash = self.cal_string_hash(f"{_neig},{_type},{_mode}")
                vid_multi_version_hash = (vid_multi_version_hash * 31 + _hash) % 2**64
            
            vid_multi_version_hash = (vid_multi_version_hash * 31 + self.cal_string_hash(str(self.in_degree(version_node_vid)))) % (2**64)
            vid_multi_version_hash = (vid_multi_version_hash * 31 + self.cal_string_hash(str(self.out_degree(version_node_vid)))) % (2**64)

            if vid_multi_version_hash_old is not None and vid_multi_version_hash != vid_multi_version_hash_old:
                if len(version_node_vid_list) == 1:
                    self.remove_node(version_node_vid_list[0])
                else:
                    reduced_central_node, sorted_vids = self.cal_reduced_version_node(version_node_vid_list, vid)

                    # delete
                    for reduced_version_node_vid in reduced_central_node:
                        for _neig, _key, _ts, _type, _mode in vid_multi_version[version_node_vid2key[reduced_version_node_vid]]:
                            if _mode == 'in':
                                self.remove_edge(_neig, vid, key=_key) if self.has_edge(_neig, vid, key=_key) else None
                            else:
                                self.remove_edge(vid, _neig, key=_key) if self.has_edge(vid, _neig, key=_key) else None
                    for sorted_vid in version_node_vid_list:
                        self.remove_node(sorted_vid)
                
                vid_multi_version_hash_old = None
                version_node_vid_list = []
                version_node_vid2key = {}

            version_node_vid_list.append(version_node_vid)
            version_node_vid2key[version_node_vid] = version_key
            vid_multi_version_hash_old = vid_multi_version_hash
        
        if len(version_node_vid_list) == 1:
            self.remove_node(version_node_vid_list[0])
        else:
            reduced_central_node, sorted_vids = self.cal_reduced_version_node(version_node_vid_list, vid)

            # delete
            for reduced_version_node_vid in reduced_central_node:
                
                for _neig, _key, _ts, _type, _mode in vid_multi_version[version_node_vid2key[reduced_version_node_vid]]:
                    if _mode == 'in':
                        self.remove_edge(_neig, vid, key=_key) if self.has_edge(_neig, vid, key=_key) else None
                    else:
                        self.remove_edge(vid, _neig, key=_key) if self.has_edge(vid, _neig, key=_key) else None
            for sorted_vid in version_node_vid_list:
                self.remove_node(sorted_vid)
    
    def cal_reduced_version_node(self, vids: List, raw_vid: int) -> List:
        """
        Returns the central nodes to be merged

        vids is a set of central nodes with identical attributes and neighborhood
        """
        vids_out_edge = {}  # {vid1: [[neig1, key1, ts1], [neig2, key2, ts2]], vid2: [], ...}
        vids_in_edge = {}
        for _vid in vids:
            vids_out_edge[_vid] = []
            for u, v, key, data in self.out_edges(_vid, keys=True, data=True):
                if u != v and v != raw_vid and v not in vids:
                    vids_out_edge[_vid].append([v, key, data.get('ts')])
            vids_out_edge[_vid] = sorted(vids_out_edge[_vid], key=lambda x: x[2])

            vids_in_edge[_vid] = []
            for u, v, key, data in self.in_edges(_vid, keys=True, data=True):
                if u != v and u != raw_vid and u not in vids:
                    vids_in_edge[_vid].append([u, key, data.get('ts')])
            vids_in_edge[_vid] = sorted(vids_in_edge[_vid], key=lambda x: x[2])
        
        in_neighbors = set(self.predecessors(vids[0])) - set(vids)
        in_neighbors = in_neighbors - {raw_vid}
        out_neighbors = set(self.successors(vids[0])) - set(vids)
        out_neighbors = out_neighbors - {raw_vid}

        in_neighbors_in_edge_ts = {}  # {neig1: [ts1, ts2], neig2: []..}
        out_neighbors_out_edge_ts = {}

        for _in_neig in in_neighbors:
            edges = self.in_edges(_in_neig, keys=True, data=True)
            in_neighbors_in_edge_ts[_in_neig] = []
            for u, v, key, data in edges:
                if u not in vids and u != raw_vid:
                    in_neighbors_in_edge_ts[_in_neig].append(data.get('ts'))
            in_neighbors_in_edge_ts[_in_neig] = sorted(in_neighbors_in_edge_ts[_in_neig])

        for _out_neig in out_neighbors:
            edges = self.out_edges(_out_neig, keys=True, data=True)
            out_neighbors_out_edge_ts[_out_neig] = []
            for u, v, key, data in edges:
                if v not in vids and v != raw_vid:
                    out_neighbors_out_edge_ts[_out_neig].append(data.get('ts'))
            out_neighbors_out_edge_ts[_out_neig] = sorted(out_neighbors_out_edge_ts[_out_neig])

        preserved_nodes = set()
        sorted_vids = None
        for i in range(len(vids_out_edge[vids[0]])):
            merged_nodes = []  # [[vid1, ts1], [vid2, ts2], ...]
            for vid, values in vids_out_edge.items():
                merged_nodes.append([vid, values[i][2]])
            preserved_i_nodes = self.cal_preserved_object(merged_nodes, out_neighbors_out_edge_ts[vids_out_edge[vids[0]][i][0]])
            preserved_nodes.update(preserved_i_nodes)

            if sorted_vids is None:
                merged_nodes.sort(key=lambda x: x[1])
                sorted_vids = [x[0] for x in merged_nodes]
        
        for i in range(len(vids_in_edge[vids[0]])):
            merged_nodes = []  # [[vid1, ts1], [vid2, ts2], ...]
            for vid, values in vids_in_edge.items():
                merged_nodes.append([vid, values[i][2]])
            preserved_i_nodes = self.cal_preserved_object(merged_nodes, in_neighbors_in_edge_ts[vids_in_edge[vids[0]][i][0]])
            preserved_nodes.update(preserved_i_nodes)

            if sorted_vids is None:
                merged_nodes.sort(key=lambda x: x[1])
                sorted_vids = [x[0] for x in merged_nodes]

        # solitary node
        if sorted_vids is None:
            return set(vids) - set([vids[0]]), vids

        return set(vids) - preserved_nodes, sorted_vids
    

    

