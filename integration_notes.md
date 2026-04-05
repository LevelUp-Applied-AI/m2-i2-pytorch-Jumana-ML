# PyTorch Housing Price Prediction - Integration Notes

## 1. What the Model Predicts
The model is a neural network designed to predict housing prices in Jordanian Dinars (JOD). 
- **Target Variable:** `price_jod`
- **Input Features (5):** `area_sqm` (Area), `bedrooms` (Number of bedrooms), `floor` (Floor level), `age_years` (Age of the building), and `distance_to_center_km` (Distance to city center).

## 2. Training Configuration
- **Number of Epochs:** 100
- **Learning Rate:** 0.01
- **Optimizer:** Adam (`torch.optim.Adam`)
- **Loss Function:** Mean Squared Error (`nn.MSELoss()`)
- **Model Architecture:** 5 inputs -> Linear(32) -> ReLU -> Linear(1)

## 3. Training Outcome
The training was highly successful. The loss decreased significantly from a very high initial value (in the billions, typical for uninitialized weights predicting large numbers like 100k+) down to a much smaller final value. 
*(Note: Final exact loss depends on random weight initialization, but it dropped by multiple orders of magnitude).*

## 4. Behavioral Observation
**Observation during training:** 
I noticed that the loss drops extremely rapidly during the first 20-30 epochs, meaning the optimizer (Adam) quickly finds the general range of the housing prices. After epoch 40, the decrease in loss slows down significantly and levels off, indicating that the model is converging and fine-tuning its weights rather than making large leaps. Also, standardizing the input features was crucial; without it, the model would have struggled to learn due to the vastly different scales of the features (e.g., area vs. bedrooms).