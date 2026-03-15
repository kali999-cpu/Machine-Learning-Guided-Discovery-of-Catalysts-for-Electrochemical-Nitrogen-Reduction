import torch
from data_utils import get_initial_dft_dataset, get_candidate_pool
from model import SurrogateHamiltonianGNN
from train import train_model
from screen import virtual_screen
from active_learning import top_candidate_acquisition, experimental_oracle

def main():
    print("=" * 60)
    print("  CATALYST SCREENING & ACTIVE LEARNING PIPELINE  ")
    print("=" * 60)
    
    # 1. Start with initial known labels (~10K, mocked as 1000 for speed)
    training_data = get_initial_dft_dataset(num_samples=1000)
    
    # Define our massive pool of uncharted configurations (mocked as 5000)
    # The active learning loop will continuously pull from this pool
    candidate_pool = get_candidate_pool(num_samples=5000)
    
    # 2. Initialize the Surrogate Hamiltonian GNN ML Model
    # Input: 5 atom features, Output: 4 properties (U_L, Selectivity, Stability, Cost)
    print("\nInitializing PyTorch Geometric Surrogate Hamiltonian GNN...")
    model = SurrogateHamiltonianGNN(in_channels=5, hidden_channels=64, out_channels=4)
    
    # Active Learning Iterations
    AL_ITERATIONS = 3
    CANDIDATES_PER_ITER = 20
    
    for iteration in range(AL_ITERATIONS):
        print("\n" + "*" * 40)
        print(f"  ACTIVE LEARNING LOOP - ITERATION {iteration + 1} / {AL_ITERATIONS}")
        print("*" * 40)
        
        # 3. Train surrogate ML model on the known labelled valid data
        print(f"\n[Step A] Training Surrogate Hamiltonian on {len(training_data)} verified samples...")
        model = train_model(model, training_data, epochs=5, batch_size=32)
        
        # 4. High-Throughput Virtual Screening via trained GNN Surrogate
        print(f"\n[Step B] High-Throughput Screening across unexplored candidates...")
        # (We only screen a random subset in full scale AL to save RAM, but here we screen the whole pool)
        screened_results = virtual_screen(model, candidate_pool, batch_size=256)
        
        # 5. Acquire Top Candidates based on Performance + Uncertainty
        print(f"\n[Step C] Identifying top candidates via Acquisition Function...")
        # Beta hyper-param increases exploration of highly uncertain structures over time loops
        top_candidates = top_candidate_acquisition(screened_results, top_n=CANDIDATES_PER_ITER, beta=1.0 + (iteration*0.5))
        
        # 6. Experimental Validation (DFT/Lab Simulation)
        print(f"\n[Step D] True Hamiltonian Simulation (Oracle)...")
        verified_data = experimental_oracle(top_candidates)
        
        # 7. Feed back into Model Training Loop
        print(f"\n[Step E] Feeding {len(verified_data)} new experimental structures back into training dataset!")
        training_data.extend(verified_data)
        
        # 8. Remove evaluated candidates from the massive uncharted pool
        evaluated_ids = [c["id"] for c in top_candidates]
        candidate_pool = [c for c in candidate_pool if getattr(c, 'id', '') not in evaluated_ids]
        
    print("\n=======================================================")
    print(" Active Learning Loop Terminated.")
    print(f" Surrogate model was continually refined. Final training set size: {len(training_data)}")
    print(" Deployed surrogate can now rapidly score billions of candidates per day.")
    print("=======================================================")

if __name__ == "__main__":
    main()
