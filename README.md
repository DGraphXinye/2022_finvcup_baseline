# 第七届信也科技杯baseline
这是第七届信也科技杯-欺诈用户风险识别的baseline。    
请在比赛网站上下载"初赛数据集.zip"文件，将zip文件中的"phase1_gdata.npz"放到路径'./xydata/raw'中。  
baseline代码中对"phase1_gdata.npz"的train_mask，随机按照7/3的比例将其划分为train/valid dataset。
 

## Environments
Implementing environment:  
- numpy = 1.21.2  
- pytorch = 1.6.0  
- torch_geometric = 1.7.2  
- torch_scatter = 2.0.8  
- torch_sparse = 0.6.9  

- GPU: Tesla V100 32G  


## Training

- **MLP**
```bash
python train.py --model mlp  --epochs 200 --device 0
python inference.py --model mlp --device 0
```

- **GCN**
```bash
python train.py --model gcn  --epochs 200 --device 0
python inference.py --model gcn --device 0
```

- **GraphSAGE**
```bash
python train.py --model sage  --epochs 200 --device 0
python inference.py --model sage --device 0
```

- **GraphSAGE (NeighborSampler)**
```bash
python train_mini_batch.py --model sage_neighsampler --epochs 200 --device 0
python inference_mini_batch.py --model sage_neighsampler --device 0
```

- **GAT (NeighborSampler)**
```bash
python train_mini_batch.py --model gat_neighsampler --epochs 200 --device 0
python inference_mini_batch.py --model gat_neighsampler --device 0
```

- **GATv2 (NeighborSampler)**
```bash
python train_mini_batch.py --model gatv2_neighsampler --epochs 200 --device 0
python inference_mini_batch.py --model gatv2_neighsampler --device 0
```

## Results:
在以上的依赖环境中，baseline中几个模型效果如下：

| Methods   | Train AUC  | Valid AUC  | Test AUC  |
|  :----  | ----  |  ---- | ---- |
| MLP | 0.7305 | 0.7328 | 0.7283 |
| GCN | 0.7272  | 0.7336 | 0.7333 |
| GraphSAGE| 0.7799 | 0.7798 | 0.7727 |
| GraphSAGE (NeighborSampler)  | 0.7916 | 0.7875 | **0.7810** |
| GAT (NeighborSampler)        | 0.7462 | 0.7411 | 0.7329 |
| GATv2 (NeighborSampler)      | 0.7818 | 0.7804 | 0.7733 |
