from Graph import *
from model import *
from torch.nn import Linear
import pandas as pd
import torch, numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler 
from torch_geometric.data import Data
from torch.nn import Linear, Dropout, Parameter, ModuleList, BatchNorm1d
from torch_geometric.data import Data, DataLoader as torch_DataLoader
import torch.nn.functional as F 
import torch.nn as nn
from torch_geometric.nn.conv  import MessagePassing,TransformerConv,GINEConv
from torch_geometric.utils    import softmax
from torch_geometric.nn       import global_add_pool as gdp
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm  
torch.cuda.empty_cache()
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

dataset = CIFData("./data/cif")

data = pd.read_csv("data/cif/id_prop.csv",header=None)
idx_list   = list(range(len(data)))
#print(idx_list)
random_num = 11
batch_size = 32	

data_set = Graph_list(idx_list, battery_dataset=dataset)
#print([i for i in data_set])

def save_graph_list_as_npz(graph_list, filename, dataset_type='data_set'):
    data_list = [k for k in graph_list] 
    #print(data_list)
    npz_data = {}
    for i, data in tqdm(enumerate(data_list), total=len(data_list), desc="Saving data"):
        npz_data[f'x_{i}'] = data.x.numpy()
        npz_data[f'edge_index_{i}'] = data.edge_index.numpy()
        npz_data[f'edge_attr_{i}'] = data.edge_attr.numpy()
        npz_data[f'y_{i}'] = data.y.numpy()
        npz_data[f'the_idx_{i}'] = data.the_idx
        #print(f'the_idx_{i}', data.the_idx)
    np.savez(filename, **npz_data)


save_graph_list_as_npz(data_set, 'graph_data_set.npz', dataset_type='data_set')