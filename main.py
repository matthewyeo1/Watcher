from transformer import train_and_predict
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    train_and_predict()
