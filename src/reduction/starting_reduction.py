import sys
import os
import datetime
import argparse
from tqdm import tqdm

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../.."))
sys.path.insert(0, project_root)
from src.utils import config
from src.utils.provgraph import DPGraph
from src.utils.opreventmodel import OPREventModel
from src.utils.datasaver import OpremSaveStrategy


def get_daily_filepath(folder_path):
    txt_files = [
        f for f in os.listdir(folder_path)
        if f.endswith('.txt')
    ]
    txt_files.sort()
    full_paths = [os.path.join(folder_path, f) for f in txt_files]
    return full_paths

def reduce_in_windows_and_save(provgraph: DPGraph, gs: OpremSaveStrategy):
    provgraph.central_node_reduction()
    provgraph.non_central_node_reduction()
    provgraph.duplicative_event_reduction()
    provgraph.multi_duplicative_event_reduction()
    gs.save_provgraph(provgraph)
    provgraph.clear_processed_nodes()


def start_reduction(daily_split_dir: str, reduced_data_dir: str, all_datas=None):
    
    # max_event_num = 100000

    daily_file_list = get_daily_filepath(daily_split_dir)
    provgraph = DPGraph(central_node_type=config.CENTRAL_NODE_TYPE)
    for daily_file in daily_file_list:
        if all_datas is not None and os.path.basename(daily_file).split('.')[0] not in all_datas:
            continue
        new_time = None
        old_time = None
        win_event_num = 0
        
        
        gs = OpremSaveStrategy(reset=False, oprem_path=os.path.join(reduced_data_dir, os.path.basename(daily_file)))
        print(f'Start reducing: {daily_file}')
        with open(daily_file, 'r', encoding='utf-8') as fin:
            for line in tqdm(fin):
                oprem = OPREventModel()
                oprem.update_from_loprem(line.strip().split('\t'))
                if oprem['e']['type'] == 'EVENT_OPEN':
                    continue
                provgraph.add_from_OPREM(oprem)
                win_event_num += 1

                standard_time = datetime.datetime.fromtimestamp(oprem['e']['ts'] / 1e9)
                new_time = standard_time
                if old_time is None:
                    old_time = standard_time
                    
                # if (new_time - old_time).total_seconds() > 900: # 15min
                if win_event_num >= 10000:
                    reduce_in_windows_and_save(provgraph, gs)
                    old_time = standard_time
                    win_event_num = 0
            reduce_in_windows_and_save(provgraph, gs)


def reducing_graph(dataset_name, all_datas=None):
    daily_split_dir = os.path.join(config.artifact_dir, 'daily', dataset_name)
    if not os.path.isdir(daily_split_dir) or not os.listdir(daily_split_dir):
        print(f"Folder does not exist or is empty: {daily_split_dir}")
        sys.exit(1)
    
    reduced_data_dir = os.path.join(config.artifact_dir, 'reduce', dataset_name)
    os.makedirs(reduced_data_dir, exist_ok=True)
    if not os.listdir(reduced_data_dir):
        start_reduction(daily_split_dir, reduced_data_dir, all_datas)


if __name__ == '__main__':
    reducing_graph("e3cadets")