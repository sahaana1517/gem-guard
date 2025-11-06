# train.py
# Jewelry Authenticity Verification - Deep Learning Model

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -------------------- 1️⃣ SETUP --------------------
# ✅ Option 1 (recommended): relative path
data_dir = "data"

# ✅ Option 2 (use this instead if your data folder is elsewhere)
# data_dir = r"C:\Users\gayet\OneDrive\Desktop\DNA PROJECT\DNA_PROJECT_GEM_GUARD\data"

# Detect device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

# -------------------- 2️⃣ DATA PREPROCESSING --------------------
# Define transformations
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load datasets
train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
val_data = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tfms)

# Data loaders
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

print(f"✅ Training images: {len(train_data)} | Validation images: {len(val_data)}")

# -------------------- 3️⃣ MODEL SETUP --------------------
# Load pretrained ResNet18
model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze earlier layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier for binary output
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 classes: authentic / fake

model = model.to(device)

# -------------------- 4️⃣ TRAINING SETUP --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# -------------------- 5️⃣ TRAINING LOOP --------------------
epochs = 10
best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation step
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"💾 Best model saved (Val Acc: {best_val_acc:.2f}%)")

print("🎯 Training complete!")
print(f"✅ Best Validation Accuracy: {best_val_acc:.2f}%")
print("📁 Model saved as best_model.pth")
