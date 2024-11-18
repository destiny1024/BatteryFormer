from torch.nn import Linear
import pandas as pd
import torch, numpy as np
import torch.optim as optim
from torch.nn import Linear, ModuleList, BatchNorm1d
import torch.nn.functional as F 
import torch.nn as nn
from torch_geometric.nn.conv  import TransformerConv
from torch_geometric.nn       import global_add_pool as gdp
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class TransformerGNN(torch.nn.Module):
    def __init__(self, heads, neurons=64, n_layers=3, dropout_rate=0.2):
        super(TransformerGNN, self).__init__()

        self.n_heads = heads
        self.n_layers = n_layers
        self.neurons = neurons
        self.neg_slope = 0.2

        # Embedding layers for node features and edge features
        self.embed_n = Linear(87, neurons)
        self.embed_e = Linear(41, neurons)

        self.dropout_rate = dropout_rate

        # Initialize lists for convolutional, linear, and batch normalization layers
        self.conv_layers = ModuleList([])
        self.lin_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(neurons, neurons, heads=heads, edge_dim=neurons))
            self.lin_layers.append(Linear(neurons * heads, neurons))  # 注意这里的输入维度调整
            self.bn_layers.append(BatchNorm1d(neurons))

        # Final linear layer for combining features
        self.linear3 = Linear(neurons, neurons)  # 调整输入维度

    def forward(self, x, edge_index, edge_attr, batch):
        # Embedding for node features and edge features
        x = self.embed_n(x)
        edge_attr = F.elu(self.embed_e(edge_attr), self.neg_slope)

        #print("Initial node features shape:", x.shape)
        #print("Initial edge features shape:", edge_attr.shape)

        # Pass through each layer
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr=edge_attr)
            x = F.elu(self.lin_layers[i](x))
            x = self.bn_layers[i](x)
            x = F.relu(x)
            #print(f"Shape after layer {i+1}: {x.shape}")

        # Global add pooling
        x = gap(x, batch)

        #print("Shape after global add pooling:", x.shape)

        # Pass through final linear layer
        x = self.linear3(x)

        #print("Shape after final linear layer:", x.shape)

        return x


class REACTION_PREDICTOR(torch.nn.Module):
    def __init__(self,inp_dim,module1,neurons=64):
        super(REACTION_PREDICTOR, self).__init__()
        self.neurons        = neurons
        self.neg_slope      = 0.2  
        self.gnn            = module1
        self.layer1         = Linear(inp_dim,neurons)
        self.layer2         = Linear(neurons,neurons)
        self.output         = Linear(neurons,1)     

    def forward(self, data0):
        x0 = data0.x
        edge_index0 = data0.edge_index
        edge_attr0 = data0.edge_attr
        batch0 = data0.batch
        
        output_feature = self.gnn(x0, edge_index0, edge_attr0, batch0)
        x = F.leaky_relu(self.layer1(output_feature), self.neg_slope)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = F.leaky_relu(self.layer2(x), self.neg_slope)
        x = F.dropout(x, p=0.3, training=self.training)
        
        y = F.leaky_relu(self.output(x), self.neg_slope).squeeze(-1)

        return y, output_feature