import torch

def top_candidate_acquisition(screened_results, top_n=20, beta=1.0):
    """
    Acquisition Function for Active Learning.
    Selects the best candidates based on a trade-off between:
    - Expected Performance (e.g. low U_L + low cost + high stability)
    - Uncertainty (Beta coefficient controls exploration vs exploitation)
    """
    print(f"Applying Acquisition Function to select Top {top_n} candidates...")
    
    # We want to MINIMIZE the score
    # Score = (predicted U_L + 0.1 * predicted cost) - (beta * uncertainty)
    # This means we EXPLORE candidates the model is highly uncertain about (High std dev)
    # And EXPLOIT candidates the model thinks are great (Low U_L / Cost)
    
    for res in screened_results:
        # Example combination of multi-objective properties
        # Lower U_L is better, Higher Selectivity is better (negative), etc.
        perf_score = (res["pred_U_L"] * 1.0) - (res["pred_selectivity"] * 0.5) + (res["pred_cost"] * 0.2)
        
        # Upper Confidence Bound (UCB) approach
        acquisition_score = perf_score - (beta * res["uncertainty"])
        res["acquisition_score"] = acquisition_score
        
    # Sort by the best Acquisition Score (Lowest)
    screened_results.sort(key=lambda x: x["acquisition_score"])
    
    # Return Top N
    top_candidates = screened_results[:top_n]
    
    print("\n--- TOP AL CANDIDATES ---")
    for idx, c in enumerate(top_candidates):
        print(f"  {idx+1}. {c['id']} | Acq Score: {c['acquisition_score']:.3f} | U_L: {c['pred_U_L']:.2f} | Uncert: {c['uncertainty']:.3f}")
        
    return top_candidates

def experimental_oracle(top_candidates):
    """
    Simulates the 'Experimental Synthesis & Testing' 
    OR running the True Quantum Mechanics (True Hamiltonian via DFT).
    Returns the True Labels so they can be added to the training set.
    """
    print(f"\nSending {len(top_candidates)} raw candidates to Experimental Oracle (True Hamiltonian)...")
    
    new_verified_data = []
    
    for cand in top_candidates:
        # Extract the underlying graph structure 
        data_obj = cand["data_obj"]
        
        # The oracle tests the structure and finds the 'True' labels
        # (Mocked here as random but normally this would take days per DFT calculation!)
        true_y = torch.rand((1, 4)) 
        
        # Attach true labels
        data_obj.y = true_y
        
        print(f"  Oracle evaluated {cand['id']}. True U_L: {true_y[0][0]:.3f}. Verified!")
        new_verified_data.append(data_obj)
        
    return new_verified_data
