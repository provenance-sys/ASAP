import sys
import os
import datetime
import argparse
from tqdm import tqdm
from torch_geometric.loader import DataLoader

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../.."))
sys.path.insert(0, project_root)
from src.utils import config
from src.dataloader.provdataloader import ProvDataset
from src.dataloader.build_word2vec import ContentEmbedder


def get_word2vec(args, raw_data_dir, trained_feat_dir):
    embedder = ContentEmbedder(dataset_name=args.dataset, 
                            train_list=args.dataset_split['train'],
                            vector_size=args.word2vec['vector_size'],
                            min_count=args.word2vec['min_count'])
    embedder.train(data_dir=raw_data_dir, model_save_dir=trained_feat_dir)
    return embedder

def dataloadering(args):
    dataloader_dir = os.path.join(config.artifact_dir, 'dataloader', args.dataset)
    os.makedirs(dataloader_dir, exist_ok=True)

    raw_data_dir = os.path.join(config.artifact_dir, 'reduce', args.dataset)
    if not os.path.isdir(raw_data_dir) or not os.listdir(raw_data_dir):
        print(f"Folder does not exist or is empty: {raw_data_dir}")
        sys.exit(1)
    
    trained_feat_dir = os.path.join(config.artifact_dir, 'trained_feat', args.dataset)
    os.makedirs(trained_feat_dir, exist_ok=True)
    trained_feat_embedder = get_word2vec(args, raw_data_dir, trained_feat_dir)

    dataloader = {}
    for _name, _list in args.dataset_split.items():
        dataset = ProvDataset(dataset_name=args.dataset,
                            splite_name=_name,  # eg. 'train'
                            splite_list=_list,  # eg. ['2018-04-02', '2018-04-03']
                            data_dir=raw_data_dir,
                            save_dir=dataloader_dir,
                            node_feature_generator=trained_feat_embedder,
                            node_content_dim=args.word2vec['vector_size'],
                            subgraph_radius=args.subgraph_radius)
        print(f"{args.dataset} {_name} have been loaded, len: {dataset.__len__()}")

        if 'train' in _name:
            dataloader[_name] = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
        else:
            dataloader[_name] = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False)
    return dataloader


if __name__ == '__main__':
    pass



