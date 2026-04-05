"""
Integration 2 — PyTorch: Housing Price Prediction
Module 2 — Programming for AI & Data Science

Complete each section below. Remove the TODO: comments and pass statements
as you implement each section. Do not change the overall structure.

Before running this script, install PyTorch:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# ─── Model Definition ─────────────────────────────────────────────────────────

class HousingModel(nn.Module):
    """Neural network for predicting housing prices from property features.

    Architecture: Linear(5, 32) -> ReLU -> Linear(32, 1)
    """

    def __init__(self):
        """Define the model layers."""
        super().__init__()
        # Define three layers as attributes:
        # 5 input features → 32 hidden units
        self.layer1 = nn.Linear(5, 32)
        # Activation function
        self.relu = nn.ReLU()
        # 32 hidden → 1 output (price prediction)
        self.layer2 = nn.Linear(32, 1)

    def forward(self, x):
        """Define the forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 5).

        Returns:
            torch.Tensor: Predictions of shape (N, 1).
        """
        # Pass x through layer1, then relu, then layer2
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        
        # Return the output
        return x


# ─── Main Training Script ─────────────────────────────────────────────────────

def main():
    """Load data, train HousingModel, and save predictions."""

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    # Load data/housing.csv using pd.read_csv
    df = pd.read_csv('data/housing.csv')
    
    # Print the shape of the DataFrame
    print(f"DataFrame shape: {df.shape}")

    # ── 2. Separate Features and Target ──────────────────────────────────────
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    
    X = df[feature_cols]
    # use double brackets to keep shape (N, 1)
    y = df[['price_jod']]   

    # ── 3. Standardize Features ───────────────────────────────────────────────
    X_mean = X.mean()
    X_std  = X.std()
    X_scaled = (X - X_mean) / X_std
    
    # Why: features have very different scales; standardization ensures
    # gradient updates are balanced across all input dimensions.

    # ── 4. Convert to Tensors ─────────────────────────────────────────────────
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values,        dtype=torch.float32)
    
    # Print X_tensor.shape and y_tensor.shape
    print(f"X_tensor shape: {X_tensor.shape}")
    print(f"y_tensor shape: {y_tensor.shape}")

    # ── 5. Instantiate Model, Loss, and Optimizer ─────────────────────────────
    model     = HousingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # ── 6. Training Loop ──────────────────────────────────────────────────────
    num_epochs = 100
    
    for epoch in range(num_epochs):
        # Forward pass: 
        predictions = model(X_tensor)
        
        # Compute loss: 
        loss = criterion(predictions, y_tensor)
        
        # Zero grads: 
        optimizer.zero_grad()
        
        # Backward: 
        loss.backward()
        
        # Update: 
        optimizer.step()
        
        # Print every 10 epochs: 
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

    # ── 7. Save Predictions ───────────────────────────────────────────────────
    # Generate predictions (wrap in torch.no_grad() for good practice)
    with torch.no_grad():
        predictions_tensor = model(X_tensor)
        
    # Convert predictions and actuals to numpy arrays
    predictions_np = predictions_tensor.numpy().flatten()
    actuals_np = y_tensor.numpy().flatten()
    
    # Build a DataFrame with columns 'actual' and 'predicted'
    results_df = pd.DataFrame({
        'actual': actuals_np,
        'predicted': predictions_np
    })
    
    # Save to predictions.csv with index=False
    results_df.to_csv('predictions.csv', index=False)
    
    # Print "Saved predictions.csv"
    print("Saved predictions.csv")


if __name__ == "__main__":
    main()
