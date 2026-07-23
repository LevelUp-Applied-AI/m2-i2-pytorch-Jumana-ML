import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import time
import matplotlib.pyplot as plt
from itertools import product
from train import HousingModel  # Import the model architecture from your train.py

def run_experiment_grid():
    # --- 1. Load and Prepare Data (Matching the logic in train.py) ---
    df = pd.read_csv('data/housing.csv')
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    X = df[feature_cols]
    y = df[['price_jod']]

    # Standardization: ensures features are on a similar scale for balanced gradient updates
    X_scaled = (X - X.mean()) / X.std()
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # --- 2. Train/Test Split (80/20) ---
    # We use a fixed seed (42) so every experiment uses the exact same split
    torch.manual_seed(42)
    indices = torch.randperm(len(X_tensor))
    split = int(0.8 * len(X_tensor))
    
    X_train, X_test = X_tensor[indices[:split]], X_tensor[indices[split:]]
    y_train, y_test = y_tensor[indices[:split]], y_tensor[indices[split:]]

    # --- 3. Hyperparameter Search Grid ---
    # Defining ranges for Learning Rate, Hidden Layer Size, and Epochs
    lrs = [0.1, 0.05, 0.01, 0.001]
    hidden_sizes = [16, 32, 64, 128]
    epochs_list = [50, 100, 200]
    
    # Generate all possible combinations
    configs = list(product(lrs, hidden_sizes, epochs_list))
    all_results = []

    print(f"Starting {len(configs)} experiments...")

    for lr, hs, epochs in configs:
        start_time = time.time()
        
        # Instantiate model with the current hidden_size config
        model = HousingModel(hidden_size=hs)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training Loop (Training on X_train/y_train only)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        training_time = time.time() - start_time

        # Evaluation (Testing on X_test/y_test which the model hasn't seen)
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test).numpy()
            test_actual = y_test.numpy()
            
            # Calculate MAE: Average error in original units (JOD)
            mae = np.mean(np.abs(test_actual - test_preds))
            
            # Calculate R²: Variance explained by the model
            ss_res = np.sum((test_actual - test_preds) ** 2)
            ss_tot = np.sum((test_actual - np.mean(test_actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

        # Record experiment data
        result = {
            "lr": lr,
            "hidden_size": hs,
            "epochs": epochs,
            "mae": float(mae),
            "r2": float(r2),
            "time_sec": float(training_time)
        }
        all_results.append(result)
        print(f"Config: LR={lr}, HS={hs}, Ep={epochs} -> Test MAE: {mae:.2f}")

    # --- 4. Log Results to JSON ---
    with open('experiments.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    # --- 5. Print Ranked Leaderboard ---
    df_res = pd.DataFrame(all_results).sort_values(by='mae')
    print("\n--- 🏆 EXPERIMENT LEADERBOARD (Top 10) ---")
    print(df_res.head(10).to_string(index=False))

    # --- 6. Produce Summary Visualization ---
    plt.figure(figsize=(10, 6))
    for hs in hidden_sizes:
        subset = df_res[df_res['hidden_size'] == hs]
        plt.scatter(subset['lr'], subset['mae'], label=f'Hidden Size {hs}')
    
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Test MAE (Lower is better)')
    plt.title('Hyperparameter Performance: LR vs MAE')
    plt.legend()
    plt.grid(True, sn="both", ls="-", alpha=0.5)
    plt.savefig('experiment_summary.png')
    print("\nSummary visualization saved as experiment_summary.png")

if __name__ == "__main__":
    run_experiment_grid()