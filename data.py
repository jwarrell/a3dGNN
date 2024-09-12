import os
import torch
import numpy as np
import GNN_core
import pickle
import random
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import networkx as nx

class graph_data(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __getitem__(self, i):
        sample = {}
        sample['node_feature'] = self.graphs[i]['x']
        sample['edges'] = self.graphs[i]['edge_index']
        sample['batch'] = self.graphs[i].batch
        return sample

    def __len__(self):
        return len(self.graphs)

    def get_degree_distance(self):
        result_degree = []
        result_distance = []
        for g in self.graphs:
            result_degree.append(GNN_core.calc_degree(g))
            result_distance.append(GNN_core.calc_distance(g))
        return result_degree, result_distance


class dataset_maker():
    # This class is to generate dataset
    def __int__(self):
        return

    def make_data(self, protein_names_file, pdb_file_dir, constrain_label_file, ratio, make_balanced,
                  output_path=None, corrupt=False):
        if constrain_label_file is not None:
            with open(constrain_label_file, 'r') as f:
                constrain_training_label = set(f.read().split('\n'))
        else:
            constrain_training_label = None

        with open(protein_names_file, 'r') as f:
            protein_names = f.read().split('\n')

        graph_dataset = []
        correct_label = []
        protein_names_exist = []
        # graph_index = 0
        for protein_index, my_protein in enumerate(protein_names):

            if os.path.exists(str(pdb_file_dir) + '/' + str(my_protein) + ".nx") \
                    and ((constrain_training_label is None) or (my_protein in constrain_training_label)):
                G = GNN_core.read_gpickle(str(pdb_file_dir) + '/' + str(my_protein) + ".nx")
                G.prot_idx = torch.tensor(protein_index, dtype=torch.long)
                G.protein_name = my_protein
                # G.graph_index = graph_index
                correct_label.append(G.y.item())
                choice = np.random.choice([0, 1, 2, 3, 4])
                if corrupt and choice == 0:
                    # G.y = 1 - G.y
                    G.x = G.x + torch.randn_like(G.x) * 0.5
                    G.corrupt = choice
                elif corrupt and choice == 1:
                    G.y = 1 - G.y
                    G.corrupt = choice
                elif corrupt:
                    G.corrupt = choice
                graph_dataset.append(G)
                protein_names_exist.append(my_protein)


        if output_path is not None:
            torch.save(correct_label, f'{output_path}/label_test.pt')
            torch.save(protein_names_exist, f'{output_path}/test_protein_names.pt')

        # ## train test partition
        # if make_balanced:
        #     graph_dataset = GNN_core.balance_dataset(graph_dataset)
        #     graph_dataset = GNN_core.alternate_dataset(graph_dataset)

        ### convert to undirect
        for index, g in enumerate(graph_dataset):
            new_edge = []
            old_edge = g['edge_index']
            old_edge = old_edge.numpy().T
            for e in old_edge:
                new_edge.append(e)
                new_edge.append([e[1], e[0]])
            g['edge_index'] = torch.from_numpy(np.array(new_edge).T)

        # breakpoint()
        # for i, g in enumerate(graph_dataset):
        #     g.graph_index = i

        if ratio is not None:
            train_data, val_data = train_test_split(graph_dataset, test_size=1 - ratio, random_state=42)
            val_data = GNN_core.balance_dataset(val_data)
            for i, g in enumerate(train_data):
                g.graph_index = i
            return graph_data(graphs=train_data), graph_data(graphs=val_data)
        else:
            return graph_data(graphs=graph_dataset)

def collate_fn(batch):
    sample = {}
    sample['node_feature'] = torch.cat([s['node_feature'] for s in batch], dim=0)
    sample['edges'] = []
    sample['batch'] = []
    for i, s in enumerate(batch):
        sample['batch'] += [i] * len(s['node_feature'])
    sample['batch'] = torch.tensor(sample['batch'])
    return sample