import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from .tokenizer import str_to_tensor

class SQLDataset(Dataset):# expands class Dataset
    """
    Custom Dataset class that loads data from a DataFrame.
    """
    def __init__(self, texts, labels):
        """
        Args:
            texts (list or Series): List of query strings.
            labels (list or Series): List of binary labels (0 or 1).
        """
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Get raw text and label
        text_raw = self.texts[idx]
        label = self.labels[idx]
        
        # 2. Vectorize text
        text_tensor = str_to_tensor(text_raw)
        
        # 3. Return tuple
        return text_tensor, torch.tensor(label, dtype=torch.long)

def get_dataloaders(csv_path, batch_size=32, test_split=0.2):
    """
    Loads the CSV, splits it into Train/Val, and returns DataLoaders(train/val).
    """
    print(f"Loading data from {csv_path}...")
    
    # Read CSV using Pandas
    df = pd.read_csv(csv_path)

    print(f"Total samples found: {len(df)}")

    # Split into Train and Validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Query'], 
        df['Label'], 
        test_size=test_split, 
        random_state=42,
        stratify=df['Label'] # equal ratio of malicious/safe in both sets
    )

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Create Dataset Objects
    train_ds = SQLDataset(train_texts, train_labels)
    val_ds = SQLDataset(val_texts, val_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader