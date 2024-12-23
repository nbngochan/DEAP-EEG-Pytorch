import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from braindecode.models import EEGNetv4
from torcheeg.datasets import DEAPDataset
from torcheeg.transforms import ToTensor

# Step 1: Define Dataset
# Define paths and parameters
data_dir = "path/to/deap/dataset"
batch_size = 64

# Load the DEAP dataset using TorchEEG's DEAPDataset API
dataset_train = DEAPDataset(root_path=data_dir, subset="train", transform=ToTensor())
dataset_test = DEAPDataset(root_path=data_dir, subset="test", transform=ToTensor())

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Step 2: Define Model
n_channels = 32  # Number of EEG channels
input_samples = 128  # Number of time samples per EEG segment

model = EEGNetv4(n_channels=n_channels, n_classes=2, input_window_samples=input_samples)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 3: Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Training Loop
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

# Step 5: Evaluation Loop
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

# Step 6: Train and Evaluate
num_epochs = 20
for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")

# Save the model
torch.save(model.state_dict(), "eegnet_deap_model.pth")
