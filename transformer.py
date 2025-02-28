import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==== Step 1: Generate Synthetic Stock Data ==== #
T = 30  # Number of past days (time steps)
num_features = 6  # Features per day (Open, Close, High, Low, Volume, Sentiment)
d = 32  # Hidden dimension for attention

# Fake stock data (batch_size=1 for simplicity)
X = torch.randn(1, T, num_features)  # Shape: (1, 30, 6)
Y = torch.randn(1, 1)  # Target for day 31 (predicting a single value)

# ==== Step 2: Feature Expansion Layer ==== #
class FeatureExpansion(nn.Module):
    def __init__(self, num_features, d):
        super().__init__()
        self.fc = nn.Linear(num_features, d)  # Expand feature dimension

    def forward(self, X):
        return self.fc(X)  # Shape: (batch_size, T, d)

# ==== Step 3: Positional Encoding ==== #
class PositionalEncoding(nn.Module):
    def __init__(self, d, T):
        super().__init__()
        self.d = d

        # Compute positional encodings for each time step (T) and feature dimension (d)
        pe = torch.zeros(T, d)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)  # Shape: (T, 1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))  # Scaling factor

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # Shape: (1, T, d), ready for batch processing

    def forward(self, X):
        return X + self.pe[:, :X.shape[1], :]  # Add positional encoding

# ==== Step 4: Self-Attention with Positional Encoding ==== #
class SelfAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)

    def forward(self, X):
        Q = self.W_Q(X)  # Shape: (batch_size, T, d)
        K = self.W_K(X)  # Shape: (batch_size, T, d)
        V = self.W_V(X)  # Shape: (batch_size, T, d)

        # Compute attention scores (Scaled Dot-Product)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
        A = F.softmax(scores, dim=-1)  # Apply softmax to get attention weights

        # Compute weighted sum (context vector)
        Z = torch.matmul(A, V)  # Shape: (batch_size, T, d)
        return Z, A

# ==== Step 5: MLP for Regression ==== #
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 64)  # Hidden layer
        self.fc2 = nn.Linear(64, 1)  # Output layer (single predicted stock value)
        self.relu = nn.ReLU()

    def forward(self, Z):
        return self.fc2(self.relu(self.fc1(Z[:, -1, :])))  # Take last time step only

# ==== Step 6: Training Setup ==== #
epochs = 100
lr = 0.001

# Initialize models
feature_expansion = FeatureExpansion(num_features, d)
pos_encoding = PositionalEncoding(d, T)
attention = SelfAttention(d)
mlp = MLP(d)

# Define loss and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    list(feature_expansion.parameters()) +
    list(attention.parameters()) +
    list(mlp.parameters()), lr=lr)

# ==== Training Loop ==== #
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients

    X_expanded = feature_expansion(X)  # Expand feature size (6 → 32)
    X_pos = pos_encoding(X_expanded)  # Apply positional encoding
    Z, A = attention(X_pos)  # Forward pass (self-attention)
    prediction = mlp(Z)  # Forward pass (MLP)

    loss = loss_fn(prediction, Y)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# ==== Final Output ==== #
print("\nFinal Prediction for Day 31:")
print(prediction.detach().numpy())  # If using GPU: prediction.cpu().detach().numpy()
