import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset.nvidia_dataset import NvidiaDataset
from model import NvidiaModel
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is ',device)
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 1e-4


CSV_PATH = './balanced_data/driving_log_balanced.csv'
IMG_DIR = 'data/IMG'
CHECKPOINT_DIR = 'checkpointsV2'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
# df = pd.read_csv(CSV_PATH,names=columns)

df = pd.read_csv(CSV_PATH)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = NvidiaDataset(train_df, IMG_DIR)
test_dataset = NvidiaDataset(test_df, IMG_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = NvidiaModel().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0

    for i,(images,angles) in enumerate(train_loader):
        images, angles = images.to(device), angles.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, angles)
        loss.backward()
        optimizer.step()

        running_loss = running_loss + loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, angles in test_loader:
            images, angles = images.to(device), angles.to(device)
            outputs = model(images)
            loss = criterion(outputs, angles)
            test_loss = test_loss + loss.item()
    
    avg_test_loss = test_loss/len(test_loader)
    test_losses.append(avg_test_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f} | Val Loss: {avg_test_loss:.4f}")
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"nvidia_model_epoch{epoch+1}.pth"))



plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.grid(True)
plt.show()