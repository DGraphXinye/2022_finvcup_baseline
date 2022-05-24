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

eval_metric = 'auc'

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


def train(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv=False):
    model.train()

    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(data.x[n_id], adjs)
        loss = F.nll_loss(out, data.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        pbar.update(batch_size)

    pbar.close()
    loss = total_loss / len(train_loader)

    return loss


@torch.no_grad()
def test(layer_loader, model, data, split_idx, device, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.eval()
    
    out = model.inference(data.x, layer_loader, device)
#     out = model.inference_all(data)
    y_pred = out.exp()  # (N,num_classes)   
    
    losses = dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        node_id = node_id.to(device)
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
            
    return losses, y_pred
        
            
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
    
    split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}
        
    data = data.to(device)
    train_idx = split_idx['train'].to(device)
        
    model_dir = prepare_folder(args.dataset, args.model)
    print('model_dir:', model_dir)

    train_loader = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[10, 5], batch_size=1024, shuffle=True, num_workers=12)
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


    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
    min_valid_loss = 1e8

    for epoch in range(1, args.epochs+1):
        loss = train(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv)
        losses, out = test(layer_loader, model, data, split_idx, device, no_conv)
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
