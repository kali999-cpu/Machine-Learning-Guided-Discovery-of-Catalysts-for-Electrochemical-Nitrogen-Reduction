import torch
from torch_geometric.data import Data, DataLoader
import random

def create_mock_candidate(idx=0):
    """
    Creates a single PyG Data object representing a mock catalyst structure.
    In real usage, this would convert an ASE Atoms object to a PyG graph.
    """
    # Random number of atoms in the structure (e.g., a surface slab + adsorbate)
    num_nodes = random.randint(10, 50)
    
    # Feature for each atom (e.g., atomic numbers, electronegativity, etc.)
    # Shape: [num_nodes, num_node_features]
    num_node_features = 5
    x = torch.rand((num_nodes, num_node_features))
    
    # Random bond connectivity (edge_index) Shape: [2, num_edges]
    # In reality, this is determined by distance cutoffs or nearest neighbors
    num_edges = random.randint(num_nodes, num_nodes * 4)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    
    # Targets: True DFT Hamiltonian properties
    # 0: Overpotential (U_L) - Lower is better
    # 1: Selectivity - Higher is better
    # 2: Stability (e.g., Formation Energy) - Lower is better
    # 3: Cost - Lower is better
    # In reality, these are None for unlabeled candidates
    y = torch.rand((1, 4))
    
    # Candidate identifier
    candidate_id = f"Candidate_{idx}"
    
    return Data(x=x, edge_index=edge_index, y=y, id=candidate_id)

def get_initial_dft_dataset(num_samples=1000):
    """
    Mocks an initial labeled dataset (e.g., ~10k DFT calculations).
    """
    print(f"Loading {num_samples} labeled 'DFT' samples into training dataset...")
    data_list = [create_mock_candidate(i) for i in range(num_samples)]
    return data_list

def get_candidate_pool(num_samples=100000):
    """
    Mocks a massive pool of untested candidates for High-Throughput Virtual Screening.
    """
    print(f"Generating massive pool of {num_samples} unverified structure candidates...")
    
    data_list = []
    for i in range(num_samples):
        # These candidates normally wouldn't have 'y' (target) defined yet
        data = create_mock_candidate(i + 1000000) 
        data.y = None # No DFT labels exist yet!
        data_list.append(data)
        
    return data_list
