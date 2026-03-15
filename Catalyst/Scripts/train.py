import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

def train_model(model, train_dataset, epochs=10, batch_size=32, lr=1e-3, device='cpu'):
    """
    Trains the Surrogate Hamiltonian using the provided "DFT" labeled dataset.
    """
    model.train()
    model.to(device)
    
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # We use Mean Squared Error to measure how far our predicted properties are from True DFT
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting Training on {len(train_dataset)} samples for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # Predict the 4 quantum properties
            outputs = model(batch.x, batch.edge_index, batch.batch)
            
            # Calculate loss against true DFT labels
            loss = criterion(outputs, batch.y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"  Epoch [{epoch+1}/{epochs}] - Loss (MSE): {epoch_loss / len(loader):.4f}")
        
    print("Training Complete. Model has learned the surrogate mappings!")
    return model
