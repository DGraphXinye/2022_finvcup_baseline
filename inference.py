# dataset name: XYGraphP1

from utils import XYGraphP1
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from models import MLP, MLPLinear, GCN, SAGE, GAT, GATv2

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd


mlp_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

gcn_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

sage_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
             }


@torch.no_grad()
def test(model, data, no_conv=False):
    # data.y is labels of shape (N, )
    model.eval()
    
    if no_conv:
        out = model(data.x)
    else:
        out = model(data.x, data.adj_t)
        
    y_pred = out.exp()  # (N,num_classes)
                
    return y_pred
        
            
def main():
    parser = argparse.ArgumentParser(description='gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='XYGraphP1')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--epochs', type=int, default=200)
    
    args = parser.parse_args()
    print(args)
    
    no_conv = False
    if args.model in ['mlp']: no_conv = True        
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = XYGraphP1(root='./', name='xydata', transform=T.ToSparseTensor())
    
    nlabels = dataset.num_classes
    if args.dataset in ['XYGraphP1']: nlabels = 2
        
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
        
    if args.dataset in ['XYGraphP1']:
        x = data.x
        x = (x-x.mean(0))/x.std(0)
        data.x = x
    if data.y.dim()==2:
        data.y = data.y.squeeze(1)        
            
    data = data.to(device)
                
    if args.model == 'mlp':
        para_dict = mlp_parameters
        model_para = mlp_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = MLP(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'gcn':   
        para_dict = gcn_parameters
        model_para = gcn_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = GCN(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'sage':        
        para_dict = sage_parameters
        model_para = sage_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = SAGE(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)

    print(f'Model {args.model} initialized')

    print(sum(p.numel() for p in model.parameters()))

    model_file = './model_files/{}/{}/model.pt'.format(args.dataset, args.model)
    print('model_file:', model_file)
    model.load_state_dict(torch.load(model_file))

    out = test(model, data, no_conv)
    
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
