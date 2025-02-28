import torch
import torch.nn as nn
import torch.nn.functional as F

# ==== Step 1: Generate Synthetic Stock Data ==== #
T = 30  # Number of past days (time steps)
num_features = 6  # Features per day (Open, Close, High, Low, Volume, Sentiment)

# Fake stock data (batch_size=1 for simplicity)
X = torch.randn(1, T, num_features)  # Shape: (batch_size, T, num_features)
Y = torch.randn(1, 1, 1)  # Target is only the next day's stock price (batch_size, 1, 1)

# ==== Step 2: Implement Self-Attention ==== #
class SelfAttention(nn.Module):
    def __init__(self, num_features, d):
        super().__init__()
        self.W_Q = nn.Linear(num_features, d, bias=False)
        self.W_K = nn.Linear(num_features, d, bias=False)
        self.W_V = nn.Linear(num_features, d, bias=False)

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

# ==== Step 3: MLP for Regression ==== #
class MLP(nn.Module):
    def __init__(self, d, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(d, 64)  # Hidden layer
        self.fc2 = nn.Linear(64, output_dim)  # Output layer
        self.relu = nn.ReLU()

    def forward(self, Z):
        return self.fc2(self.relu(self.fc1(Z)))  # Shape: (batch_size, d, output_dim)

# ==== Step 4: Training Setup ==== #
d = 32  # Hidden dimension for attention
epochs = 100
lr = 0.001

# Initialize self-attention and MLP
attention = SelfAttention(num_features, d)
mlp = MLP(d, output_dim=1)

# Define loss and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(list(attention.parameters()) + list(mlp.parameters()), lr=lr)

# ==== Training Loop ==== #
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients

    Z, A = attention(X)  # Forward pass (self-attention)

    # Pool Z to get a single vector summary of past 30 days
    Z_pooled = Z.mean(dim=1)  # Shape: (batch_size, d)

    # Predict a single value
    predictions = mlp(Z_pooled).unsqueeze(1)  # Shape: (batch_size, 1, 1)

    loss = loss_fn(predictions, Y)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# ==== Final Output ==== #
print("\nFinal Prediction for Day 31:")
print(predictions.detach().numpy())  # If using GPU: predictions.cpu().detach().numpy()
