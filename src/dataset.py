import torch
from torch_geometric.data import Dataset, DataLoader
from torch_geometric import data as DATA
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

class MoleculeDataset(Dataset):
    def __init__(self, root, dataset, csv_file, protein_graph_dict, smile_graph_dict, transform=None, pre_transform=None):
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.csv_file = csv_file
        self.protein_graph_dict = protein_graph_dict
        self.smile_graph_dict = smile_graph_dict
        self.df = pd.read_csv(self.csv_file)

        # Keep only valid indices
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            if row['smiles'] in self.smile_graph_dict and row['UniProt ID'] in self.protein_graph_dict:
                self.valid_indices.append(idx)

    def len(self):
        return len(self.valid_indices)

    def get(self, idx):
        # Use valid indices
        row = self.df.iloc[self.valid_indices[idx]]

        protein_name = row['UniProt ID']
        smile = row['smiles']

        # Get Label if available, otherwise -1 (for inference)
        if 'Label' in row:
            interaction = row['Label']
        else:
            interaction = -1.0  # Default value for inference

        # SMILES graph data
        c_size, features, edge_index = self.smile_graph_dict[smile]
        features = np.array(features)

        # Protein graph data
        c_size1, features1, edge_index1 = self.protein_graph_dict[protein_name]
        features1 = np.array(features1)

        data = DATA.Data(
            # Ligand data
            ligand_x=torch.tensor(features, dtype=torch.float),
            ligand_edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),

            # Protein data
            protein_x=torch.tensor(features1, dtype=torch.float),
            protein_edge_index=torch.tensor(edge_index1, dtype=torch.long).t().contiguous(),

            # Other information
            y=torch.tensor([interaction], dtype=torch.float),

            # Graph size information
            ligand_num_nodes=torch.tensor([features.shape[0]], dtype=torch.long),
            protein_num_nodes=torch.tensor([features1.shape[0]], dtype=torch.long)
        )

        # Store string data
        data.protein_name = protein_name
        data.smile = smile

        return data

def split_dataset(dataset, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    """Basic function to randomly split dataset"""
    torch.manual_seed(seed)

    num_samples = len(dataset)
    indices = torch.randperm(num_samples)

    train_size = int(train_ratio * num_samples)
    valid_size = int(valid_ratio * num_samples)

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size+valid_size]
    test_indices = indices[train_size+valid_size:]

    return train_indices, valid_indices, test_indices

def split_by_protein(dataset, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    """Function to split dataset by protein"""
    np.random.seed(seed)

    # Collect indices per protein (optimized: read directly from CSV)
    protein_groups = {}
    for idx in range(len(dataset)):
        # Get protein_name directly from CSV without calling dataset[idx]
        row = dataset.df.iloc[dataset.valid_indices[idx]]
        protein_name = row['UniProt ID']
        if protein_name not in protein_groups:
            protein_groups[protein_name] = []
        protein_groups[protein_name].append(idx)

    # Split proteins randomly
    proteins = list(protein_groups.keys())
    np.random.shuffle(proteins)

    n_proteins = len(proteins)
    train_size = int(train_ratio * n_proteins)
    valid_size = int(valid_ratio * n_proteins)

    train_proteins = proteins[:train_size]
    valid_proteins = proteins[train_size:train_size+valid_size]
    test_proteins = proteins[train_size+valid_size:]

    # Collect indices
    train_indices = []
    for protein in train_proteins:
        train_indices.extend(protein_groups[protein])

    valid_indices = []
    for protein in valid_proteins:
        valid_indices.extend(protein_groups[protein])

    test_indices = []
    for protein in test_proteins:
        test_indices.extend(protein_groups[protein])

    return torch.tensor(train_indices), torch.tensor(valid_indices), torch.tensor(test_indices)

def split_by_ligand(dataset, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    """Function to split dataset by ligand"""
    np.random.seed(seed)

    # Collect indices per ligand (optimized: read directly from CSV)
    ligand_groups = {}
    for idx in range(len(dataset)):
        # Get smile directly from CSV without calling dataset[idx]
        row = dataset.df.iloc[dataset.valid_indices[idx]]
        smile = row['smiles']
        if smile not in ligand_groups:
            ligand_groups[smile] = []
        ligand_groups[smile].append(idx)

    # Split ligands randomly
    ligands = list(ligand_groups.keys())
    np.random.shuffle(ligands)

    n_ligands = len(ligands)
    train_size = int(train_ratio * n_ligands)
    valid_size = int(valid_ratio * n_ligands)

    train_ligands = ligands[:train_size]
    valid_ligands = ligands[train_size:train_size+valid_size]
    test_ligands = ligands[train_size+valid_size:]

    # Collect indices
    train_indices = []
    for ligand in train_ligands:
        train_indices.extend(ligand_groups[ligand])

    valid_indices = []
    for ligand in valid_ligands:
        valid_indices.extend(ligand_groups[ligand])

    test_indices = []
    for ligand in test_ligands:
        test_indices.extend(ligand_groups[ligand])

    return torch.tensor(train_indices), torch.tensor(valid_indices), torch.tensor(test_indices)
