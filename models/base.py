import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNN(nn.Module):
    """
    Character-level CNN(for malicious query detection)
    
    1. Embedding: Converts characters into vectors.
    2. Conv1D: Scans the text for patterns.
    3. MaxPool: Finds the strongest signal (most suspicious pattern) in the text.
    4. FC layer: Makes the final decision.
    """
    def __init__(self, num_chars, embedding_dim, num_classes):
        super(CharCNN, self).__init__()
        
        # 1. embedding layer
        # Input: [Batch_Size, Sequence_Length] (Indices of characters)
        # Output: [Batch_Size, Sequence_Length, Embedding_Dim]
        self.embedding = nn.Embedding(num_chars, embedding_dim)
        
        # 2. convolutional layer
        # using a kernel size of 4 to capture 4-character patterns
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=4)
        
        # 4. fully connected layers (clasifier)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5) # dropout to prevent overfitting
        self.fc2 = nn.Linear(64, num_classes) # 2 class output (safe vs malicious)

    def forward(self, x):
        # embedding - x shape: [Batch, Seq_Len] -> [Batch, Seq_Len, Emb_Dim]
        x = self.embedding(x)
        
        # apply transpose for conv1d PyTorch Conv1d expects [Batch, Channels, Seq_Len]
        x = x.transpose(1, 2)
        
        # Convolution + ReLU Activation
        x = F.relu(self.conv1(x))
        
        # max pooling - Find the maximum value across the time dimension (Seq_Len)
        x, _ = torch.max(x, dim=2)
        
        # classification
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x