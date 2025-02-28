import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==== Step 1: Generate Synthetic Stock Data ==== #
T = 30  # Number of past days (time steps)
num_features = 5  # Features per day (Open, Close, High, Low, Volume) - sentiment separate
d = 32  # Hidden dimension for attention
num_heads = 4  # Multi-head attention

# Fake stock data (batch_size=1 for simplicity)
X = torch.randn(1, T, num_features)  # Shape: (1, 30, 5)
sentiment_score = torch.randn(1, 1)  # Random sentiment score for Day 31
Y = torch.randn(1, 1)  # Target for Day 31

# ==== Step 2: Feature Expansion ==== #
class FeatureExpansion(nn.Module):
    def __init__(self, num_features, d):
        super().__init__()
        self.fc = nn.Linear(num_features, d)

    def forward(self, X):
        return self.fc(X)

# ==== Step 3: Positional Encoding ==== #
class PositionalEncoding(nn.Module):
    def __init__(self, d, T):
        super().__init__()
        self.d = d

        # Compute positional encodings
        pe = torch.zeros(T, d)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # Shape: (1, T, d)

    def forward(self, X):
        return X + self.pe[:, :X.shape[1], :]

# ==== Step 4: Multi-Head Self-Attention ==== #
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d, num_heads):
        super().__init__()
        assert d % num_heads == 0, "d must be divisible by num_heads"
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

        Z = torch.matmul(A, V)
        Z = Z.transpose(1, 2).contiguous().view(batch_size, T, d)

        return self.W_O(Z), A

# ==== Step 5: MLP with Sentiment Integration ==== #
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

        # Learnable weight for sentiment
        self.sentiment_weight = nn.Linear(1, 1)

    def forward(self, Z, sentiment):
        stock_pred = self.fc2(self.relu(self.fc1(Z[:, -1, :])))  # Use last time step
        sentiment_adjustment = self.sentiment_weight(sentiment)  # Transform sentiment
        return stock_pred + sentiment_adjustment  # Adjust prediction

# ==== Step 6: Training Setup ==== #
epochs = 100
lr = 0.001

feature_expansion = FeatureExpansion(num_features, d)
pos_encoding = PositionalEncoding(d, T)
multihead_attention = MultiHeadSelfAttention(d, num_heads)
mlp = MLP(d)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    list(feature_expansion.parameters()) +
    list(multihead_attention.parameters()) +
    list(mlp.parameters()), lr=lr)

# ==== Training Loop ==== #
for epoch in range(epochs):
    optimizer.zero_grad()

    X_expanded = feature_expansion(X)
    X_pos = pos_encoding(X_expanded)
    Z, A = multihead_attention(X_pos)
    prediction = mlp(Z, sentiment_score)  # Pass sentiment to MLP

    loss = loss_fn(prediction, Y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# ==== Final Output ==== #
print("\nFinal Prediction for Day 31:")
print(prediction.detach().numpy())
