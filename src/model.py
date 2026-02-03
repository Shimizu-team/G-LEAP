import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, BatchNorm, LayerNorm

class BilinearAttention(nn.Module):
    """
    Bilinear Attention mechanism for protein-ligand interaction
    """
    def __init__(self, ligand_dim, protein_dim, hidden_dim=128, dropout=0.2):
        super(BilinearAttention, self).__init__()

        # Bilinear transformation matrix
        self.bilinear = nn.Bilinear(ligand_dim, protein_dim, hidden_dim, bias=True)

        # Attention scoring layers
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Final transformation layers
        self.output_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, ligand_features, protein_features):
        """
        Args:
            ligand_features: (batch_size, ligand_dim)
            protein_features: (batch_size, protein_dim)

        Returns:
            attended_features: (batch_size, hidden_dim // 2)
            attention_weights: (batch_size, 1)
        """
        # Bilinear transformation
        bilinear_output = self.bilinear(ligand_features, protein_features)  # (batch_size, hidden_dim)
        bilinear_output = self.dropout(F.relu(bilinear_output))

        # Compute attention weights
        attention_scores = self.attention_weights(bilinear_output)  # (batch_size, 1)
        attention_weights = torch.sigmoid(attention_scores)

        # Apply attention weights to bilinear features
        attended_bilinear = attention_weights * bilinear_output  # (batch_size, hidden_dim)

        # Final transformation
        output_features = self.output_transform(attended_bilinear)  # (batch_size, hidden_dim // 2)

        return output_features, attention_weights

class DualGNN_Bilinear(nn.Module):
    def __init__(self, ligand_num_features=78, protein_num_features=30, n_output=1,
                 embed_dim=32, output_dim=128, dropout=0.2, bilinear_hidden=128, num_layers=2,
                 normalization='none', pooling='mean'):
        super(DualGNN_Bilinear, self).__init__()

        self.num_layers = num_layers
        self.normalization = normalization
        self.pooling = pooling

        # Set pooling function
        if pooling == 'mean':
            self.pool_fn = global_mean_pool
        elif pooling == 'max':
            self.pool_fn = global_max_pool
        elif pooling == 'add':
            self.pool_fn = global_add_pool
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

        # Ligand GCN layers (variable number of layers)
        self.ligand_convs = nn.ModuleList()
        self.ligand_norms = nn.ModuleList()
        ligand_input_dim = ligand_num_features
        for i in range(num_layers):
            if i == num_layers - 1:  # Last layer
                out_dim = embed_dim
            else:
                out_dim = 64
            self.ligand_convs.append(GCNConv(ligand_input_dim, out_dim))

            # Add normalization layer
            if normalization == 'batch':
                self.ligand_norms.append(BatchNorm(out_dim))
            elif normalization == 'layer':
                self.ligand_norms.append(LayerNorm(out_dim))
            else:
                self.ligand_norms.append(None)

            ligand_input_dim = out_dim

        # Protein GCN layers (variable number of layers)
        self.protein_convs = nn.ModuleList()
        self.protein_norms = nn.ModuleList()
        protein_input_dim = protein_num_features
        for i in range(num_layers):
            if i == num_layers - 1:  # Last layer
                out_dim = embed_dim
            else:
                out_dim = 64
            self.protein_convs.append(GCNConv(protein_input_dim, out_dim))

            # Add normalization layer
            if normalization == 'batch':
                self.protein_norms.append(BatchNorm(out_dim))
            elif normalization == 'layer':
                self.protein_norms.append(LayerNorm(out_dim))
            else:
                self.protein_norms.append(None)

            protein_input_dim = out_dim

        # Bilinear Attention Layer
        self.bilinear_attention = BilinearAttention(
            ligand_dim=embed_dim,
            protein_dim=embed_dim,
            hidden_dim=bilinear_hidden,
            dropout=dropout
        )

        # Final prediction layers
        self.final_layers = nn.Sequential(
            nn.Linear(bilinear_hidden // 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, n_output)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize device attribute
        self.device = None

    def forward(self, data):
        # Move data to GPU
        if self.device is not None:
            data = data.to(self.device)

        # Create batch information
        ligand_batch = torch.zeros(data.ligand_x.size(0), dtype=torch.long, device=data.ligand_x.device)
        protein_batch = torch.zeros(data.protein_x.size(0), dtype=torch.long, device=data.protein_x.device)

        start_idx = 0
        for i in range(data.num_graphs):
            size = data.ligand_num_nodes[i]
            ligand_batch[start_idx:start_idx + size] = i
            start_idx += size

        start_idx = 0
        for i in range(data.num_graphs):
            size = data.protein_num_nodes[i]
            protein_batch[start_idx:start_idx + size] = i
            start_idx += size

        # Process ligand (GCN)
        x_l = data.ligand_x
        for i, conv in enumerate(self.ligand_convs):
            x_l = conv(x_l, data.ligand_edge_index)

            # Apply normalization
            if self.ligand_norms[i] is not None:
                x_l = self.ligand_norms[i](x_l)

            x_l = F.relu(x_l)
            if i < self.num_layers - 1:  # Apply dropout except for last layer
                x_l = self.dropout(x_l)
        x_l = self.pool_fn(x_l, ligand_batch)  # Pooling per batch

        # Process protein (GCN)
        x_p = data.protein_x
        for i, conv in enumerate(self.protein_convs):
            x_p = conv(x_p, data.protein_edge_index)

            # Apply normalization
            if self.protein_norms[i] is not None:
                x_p = self.protein_norms[i](x_p)

            x_p = F.relu(x_p)
            if i < self.num_layers - 1:  # Apply dropout except for last layer
                x_p = self.dropout(x_p)
        x_p = self.pool_fn(x_p, protein_batch)  # Pooling per batch

        # Bilinear Attention
        attended_features, attention_weights = self.bilinear_attention(x_l, x_p)

        # Final prediction
        output = self.final_layers(attended_features)
        output = torch.sigmoid(output)

        return output

    def get_attention_weights(self, data):
        """
        Method to get attention weights
        """
        if self.device is not None:
            data = data.to(self.device)

        # Create batch information
        ligand_batch = torch.zeros(data.ligand_x.size(0), dtype=torch.long, device=data.ligand_x.device)
        protein_batch = torch.zeros(data.protein_x.size(0), dtype=torch.long, device=data.protein_x.device)

        start_idx = 0
        for i in range(data.num_graphs):
            size = data.ligand_num_nodes[i]
            ligand_batch[start_idx:start_idx + size] = i
            start_idx += size

        start_idx = 0
        for i in range(data.num_graphs):
            size = data.protein_num_nodes[i]
            protein_batch[start_idx:start_idx + size] = i
            start_idx += size

        # Get ligand and protein features in forward pass
        x_l = F.relu(self.ligand_conv1(data.ligand_x, data.ligand_edge_index))
        x_l = self.dropout(x_l)
        x_l = F.relu(self.ligand_conv2(x_l, data.ligand_edge_index))
        x_l = global_mean_pool(x_l, ligand_batch)

        x_p = F.relu(self.protein_conv1(data.protein_x, data.protein_edge_index))
        x_p = self.dropout(x_p)
        x_p = F.relu(self.protein_conv2(x_p, data.protein_edge_index))
        x_p = global_mean_pool(x_p, protein_batch)

        # Get Bilinear attention weights
        _, bilinear_attention_weights = self.bilinear_attention(x_l, x_p)

        # Store in dictionary and return
        attention_weights = {
            'bilinear_attention': bilinear_attention_weights  # Bilinear attention
        }

        return attention_weights

