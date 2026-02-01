import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Data: MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

root = os.path.join(os.getcwd(), "data")
train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
val_ds   = datasets.MNIST(root=root, train=False, download=True, transform=transform)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

for images, labels in train_loader:
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    break

INPUT_SIZE = 28 * 28 
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
).to(DEVICE)

summary(model, (1, 28, 28))

# Training utilities
def train_one_epoch(model, loader, optimizer):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total

lr = 0.01
epochs = 10

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for ep in range(1, epochs+1):
    t_loss, t_acc = train_one_epoch(model, train_loader, optimizer)
    v_loss, v_acc = evaluate(model, val_loader)
    print(f"Epoch {ep}: train_loss={t_loss:.4f} train_acc={t_acc:.4f} | val_loss={v_loss:.4f} val_acc={v_acc:.4f}")

PATH = "model.pt"
torch.save(model.state_dict(), PATH)