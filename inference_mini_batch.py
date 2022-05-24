# dataset name: XYGraphP1

from utils import XYGraphP1
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from torch_geometric.data import NeighborSampler
from models import SAGE_NeighSampler, GAT_NeighSampler, GATv2_NeighSampler
from tqdm import tqdm

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np


sage_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

gat_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             , 'layer_heads':[4,1]
             }

gatv2_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-6
             , 'layer_heads':[4,1]
             }


@torch.no_grad()
def test(layer_loader, model, data, device, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.eval()
    
    out = model.inference(data.x, layer_loader, device)
#     out = model.inference_all(data)
    y_pred = out.exp()  # (N,num_classes)   
                
    return y_pred
        
            
def main():
    parser = argparse.ArgumentParser(description='minibatch_gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='XYGraphP1')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    print(args)
    
    no_conv = False
    if args.model in ['mlp']: no_conv = True        
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = XYGraphP1(root='./', name='xydata', transform=T.ToSparseTensor())
    
    nlabels = dataset.num_classes
    if args.dataset =='XYGraphP1': nlabels = 2
        
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
        
    if args.dataset in ['XYGraphP1']:
        x = data.x
        x = (x-x.mean(0))/x.std(0)
        data.x = x
    if data.y.dim()==2:
        data.y = data.y.squeeze(1)            
        
    data = data.to(device)
        
    layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12)        
    
    if args.model == 'sage_neighsampler':
        para_dict = sage_neighsampler_parameters
        model_para = sage_neighsampler_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = SAGE_NeighSampler(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'gat_neighsampler':   
        para_dict = gat_neighsampler_parameters
        model_para = gat_neighsampler_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = GAT_NeighSampler(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'gatv2_neighsampler':        
        para_dict = gatv2_neighsampler_parameters
        model_para = gatv2_neighsampler_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = GATv2_NeighSampler(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)

    print(f'Model {args.model} initialized')


    model_file = './model_files/{}/{}/model.pt'.format(args.dataset, args.model)
    print('model_file:', model_file)
    model.load_state_dict(torch.load(model_file))

    out = test(layer_loader, model, data, device, no_conv)

    evaluator = Evaluator('auc')
    preds_train, preds_valid = out[data.train_mask], out[data.valid_mask]
    y_train, y_valid = data.y[data.train_mask], data.y[data.valid_mask]
    train_auc = evaluator.eval(y_train, preds_train)['auc']
    valid_auc = evaluator.eval(y_valid, preds_valid)['auc']
    print('train_auc:',train_auc)
    print('valid_auc:',valid_auc)
    
    preds = out[data.test_mask].cpu().numpy()
    np.save('./submit/preds.npy', preds)


if __name__ == "__main__":
    main()
