import torch
from torch_geometric.data import DataLoader

def simulate_mc_dropout_inference(model, candidate_batch, num_passes=5):
    """
    Enables Dropout during test time and runs the same batch multiple times.
    This provides uncertainty estimates! (Monte Carlo Dropout)
    """
    model.train() # Keeping train mode so dropout continues dropping connections randomly
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_passes):
            out = model(candidate_batch.x, candidate_batch.edge_index, candidate_batch.batch)
            predictions.append(out)
            
    # Stack predictions from all passes: Shape [num_passes, batch_size, 4]
    predictions = torch.stack(predictions)
    
    # Mean prediction across all stochastic network inferences
    mean_preds = predictions.mean(dim=0)
    
    # Standard deviation: HIGHER means the model is UNCERTAIN about this catalyst!
    uncert = predictions.std(dim=0)
    
    return mean_preds, uncert

def virtual_screen(model, candidate_pool, batch_size=256, device='cpu'):
    """
    High-Throughput Virtual Screening.
    Predicts properties and uncertainties for millions of uncharted catalysts.
    """
    print(f"Commencing High-Throughput Virtual Screening of {len(candidate_pool)} unlabelled combinations...")
    
    model.to(device)
    loader = DataLoader(candidate_pool, batch_size=batch_size, shuffle=False)
    
    screened_results = []
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        
        # Calculate prediction and uncertainty via Monte Carlo dropout
        mean, std_dev = simulate_mc_dropout_inference(model, batch)
        
        # In a real workflow, candidates are mapped back to their identifiers
        # We will package up the predictions 
        for i in range(mean.shape[0]):
            
            candidate_id = batch.id[i] if hasattr(batch, 'id') else f"Cand_{batch_idx}_{i}"
            
            res = {
                "id": candidate_id,
                "data_obj": batch[i], # PyG Data representing structure
                "pred_U_L": mean[i, 0].item(),
                "pred_selectivity": mean[i, 1].item(),
                "pred_stability": mean[i, 2].item(),
                "pred_cost": mean[i, 3].item(),
                # Overall uncertainty across all predicted headers (basic mean sum)
                "uncertainty": std_dev[i].mean().item()  
            }
            screened_results.append(res)
            
    print(f"Screening complete. Ranked {len(screened_results)} candidates.")
    return screened_results
