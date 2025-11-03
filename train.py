import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from unet import UNet
from utils import HaircutDataset, draw_haircut_line
import cv2
import matplotlib.pyplot as plt
import os
import sys
import glob
from PIL import Image

# Get model name and detection type from command line arguments
model_name = sys.argv[1] if len(sys.argv) > 1 else "unet"
detection_type = sys.argv[2] if len(sys.argv) > 2 else "haircut"

# Determine folders based on model and detection type
# Format: images_UNET_haircut, masks_UNET_haircut
img_folder = f"images_{model_name.upper()}_{detection_type}"
mask_folder = f"masks_{model_name.upper()}_{detection_type}"

print(f"Training for model: {model_name.upper()}, detection type: {detection_type}")
print(f"Using image folder: {img_folder}")
print(f"Using mask folder: {mask_folder}")

# ---- Setup ----
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = HaircutDataset(img_folder, mask_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

# Load previously trained weights if they exist
model_filename = f"model_{model_name.upper()}_{detection_type}.pth"
if os.path.exists(model_filename):
    print(f"Loading pre-trained model from {model_filename}...")
    model.load_state_dict(torch.load(model_filename))

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---- Train loop ----
print("Training model...")
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for img, mask in dataloader:
        img, mask = img.to(device), mask.to(device)
        
        pred = model(img)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/10, Loss: {epoch_loss:.4f}")

# Save model with model name and detection type
model_filename = f"model_{model_name.upper()}_{detection_type}.pth"
torch.save(model.state_dict(), model_filename)
print(f"Model saved to {model_filename}")

# Predict and draw on sample
model.eval()
if len(dataset) > 0:
    sample_img, _ = dataset[0]
    with torch.no_grad():
        pred_mask = model(sample_img.unsqueeze(0).to(device))
    
    # Draw line
    orig = cv2.imread(f"{img_folder}/image_0001.png")
    if orig is not None:
        orig = cv2.resize(orig, (256, 256))
        result = draw_haircut_line(pred_mask, orig)
        cv2.imwrite(f"output_with_line_{detection_type}.jpg", result)
        print(f"Output saved to output_with_line_{detection_type}.jpg")

print("Training complete!")
