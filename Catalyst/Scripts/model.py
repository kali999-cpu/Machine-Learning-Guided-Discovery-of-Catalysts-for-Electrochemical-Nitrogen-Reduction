import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class SurrogateHamiltonianGNN(nn.Module):
    """
    Graph Neural Network serving as the Surrogate Hamiltonian.
    It maps atom structures (graphs) -> quantum mechanical properties.
    """
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=4):
        super(SurrogateHamiltonianGNN, self).__init__()
        
        # Initial atom embedding layer
        self.embedding = nn.Linear(in_channels, hidden_channels)
        
        # Message Passing (Graph Convolutional) Layers
        # Simulates information flow between bonded atoms (like electron density overlap)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Dropout for acting as Bayesian approximation (Monte Carlo Dropout)
        # This allows us to estimate the model's uncertainty during Active Learning!
        self.dropout = nn.Dropout(p=0.2)
        
        # Readout layers: Graph embeddings -> Final physical properties
        self.readout1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.readout2 = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, batch):
        """
        x: Node features [num_nodes, num_node_features]
        edge_index: Graph connectivity [2, num_edges]
        batch: Batch vector indicating atom-to-graph assignment [num_nodes]
        """
        x = self.embedding(x)
        x = torch.relu(x)
        
        # Convolution 1
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x) # Keep dropout active during inference for MC-al
        
        # Convolution 2
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # Convolution 3
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        
        # Global Pooling
        # Aggregates individual atom features to form the complete molecule/crystal feature vector
        x = global_mean_pool(x, batch)
        
        # Final property prediction
        x = self.readout1(x)
        x = torch.relu(x)
        out = self.readout2(x)
        
        return out
