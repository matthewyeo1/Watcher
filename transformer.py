import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

# ==== Step 1: Load Stock Data ==== #
print(os.getcwd())
df = pd.read_csv(r"C:\Users\akaas\PycharmProjects\Watcher\stocks_data.csv")

# Parameters
T = 30  # Time steps (past days)
num_features = 5  # Open, High, Low, Close, Volume
d = 32  # Hidden dimension for attention
num_heads = 4  # Multi-head attention
batch_size = 9  # 9 stocks

# Hardcoded sentiment analysis values
sentiment_values = torch.tensor([0.066, 0.088, 0.112, 0.153, 0.056, 0.021, 0.032, 0.07, 0.184], dtype=torch.float32)

# Select features
feature_cols = ["Open", "High", "Low", "Close", "Volume"]
stocks = df["Stock"].unique()
assert len(stocks) == batch_size, "Mismatch in batch size!"

# ==== Step 2: Prepare Sliding Window Data ==== #
X_batches, Y_batches = [], []

for stock in stocks:
    stock_df = df[df["Stock"] == stock].sort_values(by="Date")
    stock_features = stock_df[feature_cols].values

    for i in range(len(stock_features) - (T + 1)):
        X_batches.append(stock_features[i: i + T])
        Y_batches.append(stock_features[i + T][3])  # Close price

# Convert to PyTorch tensors
X_tensor = torch.tensor(np.array(X_batches), dtype=torch.float32)  # (N, 30, 5)
Y_tensor = torch.tensor(np.array(Y_batches), dtype=torch.float32).unsqueeze(1)  # (N, 1)

# Expand sentiment values to match the number of training samples
sentiment_batches = []
for stock in stocks:
    stock_df = df[df["Stock"] == stock].sort_values(by="Date")
    num_samples = len(stock_df) - (T + 1)  # Matches X and Y
    stock_index = np.where(stocks == stock)[0][0]  # Find stock index
    sentiment_batches.extend([sentiment_values[stock_index]] * num_samples)  # Repeat for exact number of samples

sentiment_tensor = torch.tensor(sentiment_batches, dtype=torch.float32).view(-1, 1)

print(f"X_tensor shape: {X_tensor.shape}")  # Should be (N, 30, 5)
print(f"Y_tensor shape: {Y_tensor.shape}")  # Should be (N, 1)
print(f"sentiment_tensor shape: {sentiment_tensor.shape}")  # Should be (N, 1)


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
        self.pe = pe.unsqueeze(0)

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
        return self.W_O(Z), A


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sentiment_weight = nn.Linear(1, 1)

    def forward(self, Z, sentiment):
        stock_pred = self.fc2(self.relu(self.fc1(Z[:, -1, :])))
        sentiment_adjustment = self.sentiment_weight(sentiment)
        return stock_pred + sentiment_adjustment


# ==== Step 4: Training Setup ==== #
epochs = 100
lr = 0.001

feature_expansion = FeatureExpansion(num_features, d)
pos_encoding = PositionalEncoding(d, T)
multihead_attention = MultiHeadSelfAttention(d, num_heads)
mlp = MLP(d)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    list(feature_expansion.parameters()) +
    list(pos_encoding.parameters()) +
    list(multihead_attention.parameters()) +
    list(mlp.parameters()), lr=lr
)

# ==== Step 5: Training Loop ==== #
for epoch in range(epochs):
    for X_batch, sentiment_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        X_expanded = feature_expansion(X_batch)
        X_pos = pos_encoding(X_expanded)
        Z, A = multihead_attention(X_pos)
        prediction = mlp(Z, sentiment_batch)
        loss = loss_fn(prediction, Y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# ==== Final Output ==== #
print("\nFinal Prediction for Next Day:")
print(prediction.detach().numpy())
