import copy
import numpy as np
import torch
import random
import os
import sys
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import pickle

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../.."))
sys.path.insert(0, project_root)
from src.utils import config
from src.dataloader.starting_dataloader import dataloadering
from src.models.glad import GLAD
from src.models.eglad import EGLAD

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_ad_score_img(img_save_path, ad_score_np):
    plt.figure(figsize=(10, 6))
    plt.hist(ad_score_np, bins=50, color='blue', alpha=0.7)
    plt.yscale('log')
    plt.xlim(0, 1)
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.savefig(img_save_path)
    plt.close()

def training_asap(args):
    torch.autograd.set_detect_anomaly(True)

    trained_asap_dir = os.path.join(config.artifact_dir, 'trained_asap', args.dataset)
    os.makedirs(trained_asap_dir, exist_ok=True)

    save_path = os.path.join(trained_asap_dir, "ASAP_MODEL" + '_pur' + str(args.pur_num) + '_mgn' + str(args.margin) + '_tc' + str(args.triplet_coef) \
                            + '_ws' + str(args.sparsity_mask_coef) + '_we' + str(args.sparsity_ent_coef) \
                            + '_dim' + str(args.hidden_dim) + '_ly' + str(args.num_hidden_layers) \
                            + '_edim' + str(args.explainer_hidden_dim) + '_ely' + str(args.explainer_num_hidden_layers) \
                            + '_lr' + str(args.lr) + '_R' + str(args.subgraph_radius))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'IMG_train/'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'IMG_val/'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'IMG_test/'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'ckpt/'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'IMG_neg/'), exist_ok=True)

    if not args.is_retraining and os.path.exists(os.path.join(save_path, 'ckpt', 'model_epoch' + str(args.asap_epochs) + '.pth')):
        print("asap model have been trained.")
        return
    
    if not args.is_retraining and os.path.exists(os.path.join(save_path, 'ckpt', 'model_epoch.pth')):
        print("asap model have been trained.")
        return

    train_logs_file_path = os.path.join(save_path, 'logs_train')
    train_log_file = open(train_logs_file_path, 'a', buffering=1)

    dataloader = dataloadering(args)
    train_loader = dataloader['train']
    val_loader = dataloader['val']
    test_loader = dataloader[args.attack]

    sample_graph, _ = next(iter(train_loader))
    node_feat_dim = sample_graph.x.shape[1]
    edge_feat_dim = sample_graph.edge_attr.shape[1]

    train_log_file.write("dataset: " + args.dataset + '\n')
    train_log_file.write(f"attack: {args.attack}\n")
    train_log_file.write(f"trian_batch_size: {args.train_batch_size}\n")
    train_log_file.write(f"test_batch_size: {args.test_batch_size}\n")
    train_log_file.write(f"epochs: {args.asap_epochs}\n")
    train_log_file.write(f"hidden_dim: {args.hidden_dim}\n")
    train_log_file.write(f"num_hidden_layers: {args.num_hidden_layers}\n")
    train_log_file.write(f"explainer_hidden_dim: {args.explainer_hidden_dim}\n")
    train_log_file.write(f"explainer_num_hidden_layers: {args.explainer_num_hidden_layers}\n")
    train_log_file.write(f"device: {args.device}\n")
    train_log_file.write(f"learning_rate: {args.lr}\n")
    train_log_file.write(f"pur_num: {args.pur_num}\n")
    train_log_file.write(f"sparsity_mask_coef: {args.sparsity_mask_coef}\n")
    train_log_file.write(f"sparsity_ent_coef: {args.sparsity_ent_coef}\n")
    train_log_file.write("node_feat_num: " + str(node_feat_dim) + ", edge_feat_dim: " + str(edge_feat_dim) + '\n')
    train_log_file.write("train_graph_num: " + str(len(train_loader.dataset)) + ", test_graph_num: " + str(len(test_loader.dataset)) + '\n')
    train_log_file.write('training...\n')
    train_log_file.write('\n')

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = EGLAD(
        input_node_dim=node_feat_dim,
        input_edge_dim=edge_feat_dim,

        encoder_hidden_dim=args.hidden_dim,
        encoder_num_hidden_layers=args.num_hidden_layers,

        explainer_hidden_dim=args.explainer_hidden_dim,
        explainer_num_hidden_layers=args.explainer_num_hidden_layers,

        sparsity_mask_coef=args.sparsity_mask_coef,
        sparsity_ent_coef=args.sparsity_ent_coef,
        pur_num=args.pur_num,
        triplet_coef=args.triplet_coef,
        margin=args.margin
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.asap_epochs,
        eta_min=1e-6
    )
    
    val_epoch2meanscore = {}
    for epoch in range(1, args.asap_epochs+1):
        loss_all = 0 
        num_sample = 0
        model.train()
        for g, neg_g in tqdm(train_loader):
            torch.cuda.empty_cache()
            g, neg_g = g.to(device), neg_g.to(device)
            optimizer.zero_grad()
            loss = model(g, neg_g)
            loss_all += loss.detach().cpu().item() * len(g.y)
            num_sample += len(g.y)

            loss.backward()
            optimizer.step()
        scheduler.step()
            
        info_train = 'Epoch {:3d}, Loss CL {:.4f}'.format(epoch, loss_all / num_sample)
        train_log_file.write(info_train + '\n')
        print(info_train)

        if epoch % 1 == 0:
            model.eval()
            all_ad_true = []
            all_ad_score = []
            for g in tqdm(test_loader):
                torch.cuda.empty_cache()
                all_ad_true.append(g.y.cpu())
                g = g.to(device)
                with torch.no_grad():
                    ano_score = model.get_anomaly_score(g)
                    all_ad_score.append(ano_score.detach().cpu())
            
            all_ad_score = torch.cat(all_ad_score)
            ad_score_np = all_ad_score.numpy()
            img_path = os.path.join(save_path, 'IMG_test/epoch_' + str(epoch) + '.png')
            save_ad_score_img(img_path, ad_score_np)
            train_log_file.write(f'test anomaly score(mean): {np.mean(ad_score_np)}\n')
            print(f'test anomaly score(mean): {np.mean(ad_score_np)}')
            if epoch > args.asap_save_epoch:
                val_epoch2meanscore[epoch] = float(np.mean(ad_score_np))

        torch.save(model.state_dict(), os.path.join(save_path, 'ckpt/model_epoch' + str(epoch) + '.pth'))
    optimal_epoch = min(val_epoch2meanscore, key=val_epoch2meanscore.get)
    pickle.dump(optimal_epoch, open(os.path.join(save_path, 'ckpt/optimal_epoch.pkl'), "wb"))
    train_log_file.write(f'\noptimal epoch: {optimal_epoch}\n')
    

if __name__ == '__main__':
    pass

