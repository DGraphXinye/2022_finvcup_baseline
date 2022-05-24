# dataset name: XYGraphP1

from utils import XYGraphP1
from utils.utils import prepare_folder
from models import MLP, MLPLinear, GCN, SAGE, GAT, GATv2

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
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


def train(model, data, train_idx, optimizer, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.train()

    optimizer.zero_grad()
    if no_conv:
        out = model(data.x[train_idx])
    else:
        out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, no_conv=False):
    # data.y is labels of shape (N, )
    model.eval()
    
    if no_conv:
        out = model(data.x)
    else:
        out = model(data.x, data.adj_t)
        
    y_pred = out.exp()  # (N,num_classes)
    
    losses = dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
            
    return losses, y_pred
        
            
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
    
    split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}
        
    data = data.to(device)
    train_idx = split_idx['train'].to(device)
        
    model_dir = prepare_folder(args.dataset, args.model)
    print('model_dir:', model_dir)
        
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

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
    min_valid_loss = 1e8

    for epoch in range(1, args.epochs+1):
        loss = train(model, data, train_idx, optimizer, no_conv)
        losses, out = test(model, data, split_idx, no_conv)
        train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), model_dir+'model.pt')

        if epoch % args.log_steps == 0:
            print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_loss:.3f}%, '
                      f'Valid: {100 * valid_loss:.3f}% '
                      f'Test: {100 * test_loss:.3f}%')


if __name__ == "__main__":
    main()
