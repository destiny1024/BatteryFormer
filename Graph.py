import os
import glob
import torch
import csv
import random
import functools
import warnings
import numpy as np
import json
import pymatgen
from pymatgen.core.structure import Structure
from pymatgen.core.structure import Element
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data

config_path = os.path.join('data','mp_config_onehot.json')
with open(config_path) as f:
    config = json.load(f)
    #print(config)

class CIFData(Dataset):
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id+'.cif'))
        atoms=crystal.atomic_numbers
        atomnum=config['atomic_numbers']
        z_dict = {z:i for i, z in enumerate(atomnum)}
        one_hotvec=np.array(config["node_vectors"])
        atom_fea = np.vstack([one_hotvec[z_dict[atoms[i]]] for i in range(len(crystal))])
        #radius, max_num_nbr = 8, 12
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_element, nbr_fea_idx, nbr_fea = [], [], []

        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr)))
                nbr_element.append([nbrs[0].specie for nbrs in nbr])
                numbers = [Element(elem).atomic_radius for elem in nbr_element[0]]
                nbr_fea.append([(numbers[i] + numbers[i+1]) / 2 for i in range(len(numbers)-1)] + [numbers[-1]]+ [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
        
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_element.append([nbrs[0].specie for nbrs in nbr[:self.max_num_nbr]])
                numbers = [Element(elem).atomic_radius for elem in nbr_element[0]]
                nbr_fea.append([(numbers[i] + numbers[i+1]) / 2 for i in range(len(numbers)-1)] + [numbers[-1]])
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        #print("nbr_fea: ",nbr_fea.shape)
        nbr_fea_idx = self.format_adj_matrix(torch.LongTensor(nbr_fea_idx))
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
    
    def format_adj_matrix(self,adj_matrix):
        size = len(adj_matrix)
        src_list = list(range(size))
        all_src_nodes = torch.tensor([[x]*adj_matrix.shape[1] for x in src_list]).view(-1).long().unsqueeze(0)
        all_dst_nodes = adj_matrix.view(-1).unsqueeze(0)
        return torch.cat((all_src_nodes,all_dst_nodes),dim=0)

class Graph_list(Dataset):
    def __init__(self,task_idx,battery_dataset):
        self.task_idx = task_idx
        self.battery_dataset = battery_dataset
    
    def __len__(self):
        return len(self.task_idx)
    
    def __getitem__(self,idx):
        i = self.task_idx[idx]
        material = self.battery_dataset[i]
        
        def Crystal_dataset(materials):
            node_features = materials[0][0]
            edge_features = materials[0][1]
            edge_features = edge_features.view(-1,41)
            adj_matrix = materials[0][2]
            y = materials[1]
            mp_id = materials[2]
            
            return node_features, edge_features, adj_matrix, y, mp_id
        
        materialss = Crystal_dataset(material)
        graph_crystal = Data(x=materialss[0],edge_attr=materialss[1],edge_index=materialss[2],y=materialss[3],the_idx=materialss[4])
        #print(graph_crystal)
        return graph_crystal

    
class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)