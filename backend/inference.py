import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class StockPredictorModel(nn.Module):
    def __init__(self):
        super(StockPredictorModel, self).__init__()
        # The original model expects 1 feature per timestep (for each stock)
        self.fc1 = nn.Linear(1, 16)   # Input: 1 feature per timestep, Output: 16
        self.fc2 = nn.Linear(16, 16)  # Input: 16, Output: 16
        self.fc3 = nn.Linear(16, 1)   # Input: 16, Output: 1 (final prediction)

    def forward(self, x, sentiment):
        # Flatten the input so that each stock's features are presented correctly to the model
        x = x.view(-1, 1)  # Flatten to (batch_size * history_window, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x, None  # Modify as needed for your model's output


# Device setup (CUDA or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained model
model_path = r'C:\Users\akaas\PycharmProjects\Watcher\stock_predictor_model.pth'
model = StockPredictorModel()
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)  # Load weights into the model
model.to(device)
model.eval()

# Read the stock data
df = pd.read_csv(r"C:\Users\akaas\PycharmProjects\Watcher\backend\stocks_data.csv")

HISTORY_WINDOW = 60  # Use 60 days of history for both training and inference
sentiment_values = torch.tensor([0.066, 0.088, 0.112, 0.153, 0.056, 0.021, 0.032, 0.07, 0.184], dtype=torch.float32)
stocks = sorted(df["Stock"].unique())
num_features = 5  # Open, High, Low, Close, Volume
normalization_params = {}

# Prepare for inference
inference_X = []
inference_sentiment = []
inference_stock_names = []

for idx, stock in enumerate(stocks):
    stock_df = df[df["Stock"] == stock].sort_values(by="Date", ascending=True)

    # Skip stocks with insufficient data for inference
    if len(stock_df) < HISTORY_WINDOW + 1:
        continue

    stock_features = np.zeros((HISTORY_WINDOW, num_features), dtype=np.float32)

    # Normalize using stored parameters (same as during training)
    for col_idx, col in enumerate(["Open", "High", "Low", "Close", "Volume"]):
        mean = stock_df[col].mean()
        std = stock_df[col].std()

        # Save the normalization parameters for later use in denormalization
        if col == "Close":
            normalization_params[stock] = {"Close": {"mean": mean, "std": std}}

        # Get the last HISTORY_WINDOW days of raw stock values
        raw_values = stock_df[col].values[-HISTORY_WINDOW:]

        # Apply normalization (same as training)
        stock_features[:, col_idx] = (raw_values - mean) / std

    # Append normalized features and sentiment to inference lists
    inference_X.append(stock_features)
    inference_sentiment.append(sentiment_values[idx].item())
    inference_stock_names.append(stock)

# Convert the normalized features and sentiment values into tensors
inference_X_tensor = torch.tensor(np.array(inference_X), dtype=torch.float32).to(device)
inference_sentiment_tensor = torch.tensor(inference_sentiment, dtype=torch.float32).view(-1, 1).to(device)

# Flatten the input tensor (adjust the shape for the model)
inference_X_tensor_flattened = inference_X_tensor.view(-1, 1)  # Flatten to (batch_size * history_window, 1)

# ==== Make Predictions ==== #
model.eval()
with torch.no_grad():
    predictions, _ = model(inference_X_tensor_flattened, inference_sentiment_tensor)
    normalized_predictions = predictions.cpu().numpy().flatten()

# Denormalize predictions to actual stock price values
final_predictions = []
for i, stock in enumerate(inference_stock_names):
    close_mean = normalization_params[stock]['Close']['mean']
    close_std = normalization_params[stock]['Close']['std']

    # Denormalize using the mean and std of the 'Close' column
    denormalized_pred = normalized_predictions[i] * close_std + close_mean
    final_predictions.append(denormalized_pred)

    # Output the prediction for each stock
    print(f"Stock: {stock}, Predicted Close: ${denormalized_pred:.2f}")

# Output the final predictions
print("Final predictions:", final_predictions)
