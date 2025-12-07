import torch
import torch.nn as nn
import torch.optim as optim
import time
from models.base import CharCNN
from utils.dataloader import get_dataloaders
from utils.tokenizer import ALPHABET
import argparse

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
EMBEDDING_DIM = 64
NUM_CHARS = len(ALPHABET) + 1  # +1 for padding

def train_model(path):
    # use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
    model = CharCNN(NUM_CHARS, EMBEDDING_DIM, num_classes=2)
    model = model.to(device)

    # Data Preparation
    train_loader, val_loader = get_dataloaders(path,BATCH_SIZE)

    # Loss and Optimizer
    # Using CrossEntropyLoss for classification, AdamW optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Training Loop
    print("\nTRAINING STARTED")
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # zero gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward pass and optimization
            loss.backward()
            optimizer.step()

            # track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # validation phase
        model.eval() # set model to evaluation mode (turns off dropout)
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad(): # No gradient needed for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Print Epoch Stats
        epoch_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

    # Save Model
    save_path = f"model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,help="dataset path",required=True)
    args = parser.parse_args()
    
    train_model(args.path)