import torch

from transformer import feature_expansion, pos_encoding, multihead_attention, MLP, load_stock_data  # Import model and function
# Reshape fixed_sentiment_score to match the input shape expected by the model
fixed_sentiment_score = torch.tensor([0.066, 0.088, 0.112, 0.153, 0.056, 0.021, 0.032, 0.07, 0.184], dtype=torch.float32)
fixed_sentiment_score = fixed_sentiment_score.unsqueeze(0)  # Reshape from (9,) to (1, 9)
fixed_sentiment_score = fixed_sentiment_score.unsqueeze(1)  # Reshape from (1, 9) to (1, 1, 9)

# Load the trained model
mlp = MLP(d=32)  # Initialize your model, make sure d is the correct input size
mlp.load_state_dict(torch.load(r'C:\Users\akaas\PycharmProjects\Watcher\backend\best_stock_model.pt'))  # Load the trained model from .pth file
mlp.eval()  # Set model to evaluation mode

# Load the stock data (replace 'your_stock_data.csv' with the actual path)
X = load_stock_data(r'C:\Users\akaas\PycharmProjects\Watcher\backend\stocks_data.csv')  # Shape: (9, 30, 5)

# Perform inference for Day 31
X_expanded = feature_expansion(X)
X_pos = pos_encoding(X_expanded)
Z, A = multihead_attention(X_pos)
prediction_31 = mlp(Z, fixed_sentiment_score)

# Convert prediction to NumPy for easy use
prediction_31_values = prediction_31.detach().numpy()

# Print the predictions for Day 31
print("\nPredictions for Day 31:")
for i, pred in enumerate(prediction_31_values):
    print(f"Stock {i+1}: {pred[0]}")

# Use the prediction for Day 31 to predict Day 32 (if desired)
prediction_31_tensor = prediction_31.unsqueeze(2).repeat(1, 1, 5)  # Shape: (9, 1, 5)
X_new = torch.cat((X[:, 1:, :], prediction_31_tensor), dim=1)  # Shift window to include Day 31 prediction

X_expanded_32 = feature_expansion(X_new)
X_pos_32 = pos_encoding(X_expanded_32)
Z_32, A_32 = multihead_attention(X_pos_32)
prediction_32 = mlp(Z_32, fixed_sentiment_score)  # Inference for Day 32

# Convert Day 32 prediction to NumPy
prediction_32_values = prediction_32.detach().numpy()

# Print the predictions for Day 32
print("\nPredictions for Day 32:")
for i, pred in enumerate(prediction_32_values):
    print(f"Stock {i+1}: {pred[0]}")

# Save predictions to a file
with open("future_predictions.txt", "w") as f:
    for i in range(9):
        f.write(f"Stock {i+1} - Day 31: {prediction_31_values[i][0]}\n")
        f.write(f"Stock {i+1} - Day 32: {prediction_32_values[i][0]}\n")

print("Predictions saved to 'future_predictions.txt'")
