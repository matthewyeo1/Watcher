import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ==== Step 1: Load Stock Data ==== #
print(os.getcwd())
df = pd.read_csv(r"C:\Users\user\Watcher\backend\stocks_data.csv")

# Parameters
HISTORY_WINDOW = 60  # Use 60 days of history for both training and inference
TRAIN_RATIO = 0.8  # Use 80% of data for training, 10% for validation, 10% for testing
num_features = 5  # Open, High, Low, Close, Volume
d = 32  # Hidden dimension for attention
num_heads = 4  # Multi-head attention
batch_size = 16  # Batch size for training

# Prepare to track normalization parameters per stock
normalization_params = {}

# Hardcoded sentiment analysis values
sentiment_values = torch.tensor([0.066, 0.088, 0.112, 0.153, 0.056, 0.021, 0.032, 0.07, 0.184], dtype=torch.float32)

# Select and sort stocks alphabetically
stocks = sorted(df["Stock"].unique())
assert len(stocks) == len(sentiment_values), "Mismatch in sentiment values and stocks!"

# ==== Step 2: Prepare Data (Avoiding Data Leakage) ==== #
X_train_all, X_val_all, X_test_all = [], [], []
y_train_all, y_val_all, y_test_all = [], [], []
sentiment_train_all, sentiment_val_all, sentiment_test_all = [], [], []


def create_sequences(data, window_size):
    """Create sequences of window_size days for training and prediction"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 3])  # 3 = Close price column index
    return np.array(X), np.array(y)


for idx, stock in enumerate(stocks):
    stock_df = df[df["Stock"] == stock].sort_values(by="Date", ascending=True)

    # Store raw features for later denormalization
    stock_features_raw = stock_df[["Open", "High", "Low", "Close", "Volume"]].values

    # Normalize each stock's data individually to prevent data leakage
    stock_features = np.zeros_like(stock_features_raw, dtype=np.float32)
    normalization_params[stock] = {}

    for col_idx, col in enumerate(["Open", "High", "Low", "Close", "Volume"]):
        mean = stock_df[col].mean()
        std = stock_df[col].std()
        normalization_params[stock][col] = {'mean': mean, 'std': std}
        stock_features[:, col_idx] = (stock_features_raw[:, col_idx] - mean) / std

    # Create sequences
    X, y = create_sequences(stock_features, HISTORY_WINDOW)

    # Skip stocks with insufficient data
    if len(X) < 10:  # At least 10 samples needed
        print(f"Skipping stock {stock} due to insufficient data (only {len(X)} sequences available).")
        continue

    # Split into train/validation/test
    # First split into train and temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=TRAIN_RATIO, shuffle=False  # Keep time order
    )

    # Then split temp into validation and test (50% each of what's left)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=False  # Keep time order
    )

    # Get sentiment value
    sentiment = sentiment_values[idx].item()

    # Add to all stocks data
    X_train_all.extend(X_train)
    X_val_all.extend(X_val)
    X_test_all.extend(X_test)

    y_train_all.extend(y_train)
    y_val_all.extend(y_val)
    y_test_all.extend(y_test)

    sentiment_train_all.extend([sentiment] * len(X_train))
    sentiment_val_all.extend([sentiment] * len(X_val))
    sentiment_test_all.extend([sentiment] * len(X_test))

X_train_tensor = torch.tensor(np.array(X_train_all), dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(np.array(y_train_all), dtype=torch.float32).view(-1, 1).to(device)
sentiment_train_tensor = torch.tensor(sentiment_train_all, dtype=torch.float32).view(-1, 1).to(device)

X_val_tensor = torch.tensor(np.array(X_val_all), dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(np.array(y_val_all), dtype=torch.float32).view(-1, 1).to(device)
sentiment_val_tensor = torch.tensor(sentiment_val_all, dtype=torch.float32).view(-1, 1).to(device)

X_test_tensor = torch.tensor(np.array(X_test_all), dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(np.array(y_test_all), dtype=torch.float32).view(-1, 1).to(device)
sentiment_test_tensor = torch.tensor(sentiment_test_all, dtype=torch.float32).view(-1, 1).to(device)

train_dataset = TensorDataset(X_train_tensor, sentiment_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, sentiment_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test_tensor, sentiment_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ==== Step 3: Define Enhanced Model Components ==== #
class FeatureExpansion(nn.Module):
    def __init__(self, num_features, d):
        super().__init__()
        self.fc = nn.Linear(num_features, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, X):
        return self.norm(F.gelu(self.fc(X)))


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=1000):
        super().__init__()
        self.d = d
  
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, X):
        seq_len = X.shape[1]
        return X + self.pe[:, :seq_len, :].to(X.device)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d, num_heads, dropout=0.1):
        super().__init__()
        assert d % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d // num_heads

        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, X):
        # Apply layer normalization first
        X_norm = self.norm(X)

        batch_size, T, d = X_norm.shape
        Q = self.W_Q(X_norm).view(batch_size, T, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_K(X_norm).view(batch_size, T, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_V(X_norm).view(batch_size, T, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        A = F.softmax(scores, dim=-1)
        A = self.dropout(A)
        Z = torch.matmul(A, V).transpose(1, 2).contiguous().view(batch_size, T, d)
        Z = self.W_O(Z)

        return X + Z, A


class FeedForward(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 4)
        self.fc2 = nn.Linear(d * 4, d)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, X):
        X_norm = self.norm(X)
        return X + self.dropout(self.fc2(F.gelu(self.fc1(X_norm))))


class SentimentMLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        return self.fc3(x)


class StockPredictionModel(nn.Module):
    def __init__(self, num_features, d, num_heads, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_expansion = FeatureExpansion(num_features, d)
        self.positional_encoding = PositionalEncoding(d)

        # Multiple layers of attention and feed-forward networks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                MultiHeadSelfAttention(d, num_heads, dropout),
                FeedForward(d, dropout)
            ]))

        # Separate paths for time series and sentiment
        self.time_series_mlp = nn.Sequential(
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.sentiment_mlp = SentimentMLP()

        # Final combination layer with a small weight for sentiment
        self.alpha = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

    def forward(self, X, sentiment):
        X = self.feature_expansion(X)
        X = self.positional_encoding(X)

        attentions = []
        for attn, ff in self.layers:
            X, attn_weights = attn(X)
            X = ff(X)
            attentions.append(attn_weights)

        time_series_pred = self.time_series_mlp(X[:, -1, :])
        sentiment_pred = self.sentiment_mlp(sentiment)

        final_pred = time_series_pred + self.alpha * sentiment_pred

        return final_pred, attentions


# ==== Step 4: Training and Evaluation Functions ==== #
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, sentiment_batch, y_batch in dataloader:
        optimizer.zero_grad()

        prediction, _ = model(X_batch, sentiment_batch)
        loss = criterion(prediction, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, sentiment_batch, y_batch in dataloader:
            prediction, _ = model(X_batch, sentiment_batch)
            loss = criterion(prediction, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            all_preds.extend(prediction.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    mse = np.mean((np.array(all_preds) - np.array(all_targets)) ** 2)

    return total_loss / len(dataloader.dataset), mae, mse, all_preds


# ==== Step 5: Model Training with Early Stopping ==== #
model = StockPredictionModel(num_features, d, num_heads).to(device)
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

best_val_loss = float('inf')
patience = 10
patience_counter = 0
epochs = 100

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_mae, val_mse, _ = evaluate(model, val_loader, criterion, device)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}, Val MSE: {val_mse:.6f}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        torch.save(model.state_dict(), 'best_stock_model.pt')
        print("Model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

model.load_state_dict(torch.load('best_stock_model.pt'))

# ==== Step 6: Final Evaluation and Inference ==== #
test_loss, test_mae, test_mse, test_predictions = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.6f}, Test MAE: {test_mae:.6f}, Test MSE: {test_mse:.6f}")

test_predictions_denormalized = []

inference_X = []
inference_sentiment = []
inference_stock_names = []

for idx, stock in enumerate(stocks):
    stock_df = df[df["Stock"] == stock].sort_values(by="Date", ascending=True)

    if len(stock_df) < HISTORY_WINDOW + 1:
        continue  

    stock_features = np.zeros((HISTORY_WINDOW, num_features), dtype=np.float32)

    # Normalize using stored parameters
    for col_idx, col in enumerate(["Open", "High", "Low", "Close", "Volume"]):
        mean = normalization_params[stock][col]['mean']
        std = normalization_params[stock][col]['std']
        raw_values = stock_df[col].values[-HISTORY_WINDOW:]
        stock_features[:, col_idx] = (raw_values - mean) / std

    inference_X.append(stock_features)
    inference_sentiment.append(sentiment_values[idx].item())
    inference_stock_names.append(stock)

inference_X_tensor = torch.tensor(np.array(inference_X), dtype=torch.float32).to(device)
inference_sentiment_tensor = torch.tensor(inference_sentiment, dtype=torch.float32).view(-1, 1).to(device)

# Make predictions
model.eval()
with torch.no_grad():
    predictions, _ = model(inference_X_tensor, inference_sentiment_tensor)
    normalized_predictions = predictions.cpu().numpy().flatten()

# Denormalize predictions to actual stock price values
final_predictions = []
for i, stock in enumerate(inference_stock_names):
    close_mean = normalization_params[stock]['Close']['mean']
    close_std = normalization_params[stock]['Close']['std']
    denormalized_pred = normalized_predictions[i] * close_std + close_mean
    final_predictions.append(denormalized_pred)
    print(f"Stock: {stock}, Predicted Close: ${denormalized_pred:.2f}")

print("Final predictions:", final_predictions)

# ==== Save the Model as .pth File ==== #
model = SentimentMLP()
torch.save(model.state_dict(), 'stock_predictor_model.pth')
print("Model saved as 'stock_predictor_model.pth'")

# ==== Save Final Output to a .txt File ==== #
with open("final_prediction.txt", "w") as f:
    f.write(f"Final Prediction for Day 31: {final_predictions}\n")
    print("Prediction saved to 'final_prediction.txt'")