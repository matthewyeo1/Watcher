import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

# Select device (GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==== Step 1: Load Stock Data and Normalize ==== #
print(os.getcwd())
df = pd.read_csv(r"stocks_history.csv")

# Parameters
T = 30  # Time steps (past days)
num_features = 5  # Open, High, Low, Close, Volume
d = 32  # Hidden dimension for attention
num_heads = 4  # Multi-head attention
batch_size = 32  # Increased batch size

# Normalize stock features (Standardization) - Using training data statistics
mean_values = df[["Open", "High", "Low", "Close", "Volume"]].mean()
std_values = df[["Open", "High", "Low", "Close", "Volume"]].std()
df[["Open", "High", "Low", "Close", "Volume"]] = (df[["Open", "High", "Low", "Close", "Volume"]] - mean_values) / std_values

# Hardcoded sentiment analysis values
sentiment_values = torch.tensor([0.066, 0.088, 0.112, 0.153, 0.056, 0.021, 0.032, 0.07, 0.184], dtype=torch.float32)

# Select features
feature_cols = ["Open", "High", "Low", "Close", "Volume"]
stocks = df["Stock"].unique()
assert len(stocks) == len(sentiment_values), "Mismatch in sentiment values and stocks!"

# ==== Step 2: Prepare Sliding Window Data ==== #
X_batches, Y_batches, sentiment_batches = [], [], []

for stock in stocks:
    stock_df = df[df["Stock"] == stock].sort_values(by="Date")
    stock_features = stock_df[feature_cols].values
    num_samples = len(stock_features) - (T + 1)
    stock_index = np.where(stocks == stock)[0][0]

    for i in range(num_samples):
        X_batches.append(stock_features[i: i + T])
        Y_batches.append(stock_features[i + T][3])  # Close price
        sentiment_batches.append(sentiment_values[stock_index])

# Convert to PyTorch tensors
X_tensor = torch.tensor(np.array(X_batches), dtype=torch.float32).to(device)  # (N, 30, 5)
Y_tensor = torch.tensor(np.array(Y_batches), dtype=torch.float32).unsqueeze(1).to(device)  # (N, 1)
sentiment_tensor = torch.tensor(sentiment_batches, dtype=torch.float32).view(-1, 1).to(device)  # (N, 1)

# DataLoader
dataset = TensorDataset(X_tensor, sentiment_tensor, Y_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ==== Step 3: Define Model Components ==== #
class FeatureExpansion(nn.Module):
    def __init__(self, num_features, d):
        super().__init__()
        self.fc = nn.Linear(num_features, d)

    def forward(self, X):
        return self.fc(X)


class PositionalEncoding(nn.Module):
    def __init__(self, d, T):
        super().__init__()
        self.d = d
        pe = torch.zeros(T, d)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).to(device)

    def forward(self, X):
        return X + self.pe[:, :X.shape[1], :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d, num_heads):
        super().__init__()
        assert d % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d // num_heads
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d)

    def forward(self, X):
        batch_size, T, d = X.shape
        Q = self.W_Q(X).view(batch_size, T, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_K(X).view(batch_size, T, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_V(X).view(batch_size, T, self.num_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        A = F.softmax(scores, dim=-1)
        Z = torch.matmul(A, V).transpose(1, 2).contiguous().view(batch_size, T, d)
        return self.W_O(Z)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sentiment_weight = nn.Linear(1, 1)

    def forward(self, Z, sentiment):
        x = self.relu(self.fc1(Z[:, -1, :]))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        stock_pred = self.fc3(x)
        sentiment_adjustment = self.sentiment_weight(sentiment)
        return stock_pred + sentiment_adjustment

# ==== Step 4: Training Setup ==== #
epochs = 150
lr = 0.0005
patience = 15  # Stop training if no improvement after 15 epochs
best_loss = float("inf")
early_stop_counter = 0

feature_expansion = FeatureExpansion(num_features, d).to(device)
pos_encoding = PositionalEncoding(d, T).to(device)
multihead_attention = MultiHeadSelfAttention(d, num_heads).to(device)
mlp = MLP(d).to(device)

loss_fn = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.Adam(list(feature_expansion.parameters()) +
                             list(pos_encoding.parameters()) +
                             list(multihead_attention.parameters()) +
                             list(mlp.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# ==== Training Loop ==== #
for epoch in range(epochs):
    epoch_loss = 0.0

    for X_batch, sentiment_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        X_expanded = feature_expansion(X_batch)
        X_pos = pos_encoding(X_expanded)
        Z = multihead_attention(X_pos)
        prediction = mlp(Z, sentiment_batch)
        loss = loss_fn(prediction, Y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    scheduler.step(epoch_loss)

    print(f"Epoch {epoch}, Loss: {epoch_loss}")

    # Early Stopping Logic
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        early_stop_counter = 0  # Reset counter when improvement is seen
    else:
        early_stop_counter += 1  # Increment counter if no improvement

    if early_stop_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# Final Predictions (Denormalized)
final_predictions = (prediction.cpu().detach().numpy() * std_values["Close"]) + mean_values["Close"]
final_predictions = final_predictions[-9:]
print("Prediction for tomorrow:")
print(final_predictions)
