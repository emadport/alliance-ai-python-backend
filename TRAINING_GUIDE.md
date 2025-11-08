# Model Training Guide

Complete guide to training your AI models with web-learned data.

## Quick Start: Train Your Model

### Option 1: Train via API

```bash
# Start the server
python3 -m uvicorn server:app --reload --port 8000

# Make training request (from frontend or curl)
POST /train?model=unet&type=line_detection

# Check training status
GET /train/status?model=unet&type=line_detection
```

### Option 2: Train via Command Line

```bash
# Train UNet for line detection
python3 train.py unet line_detection

# Train UNet for haircut detection
python3 train.py unet haircut

# Train RNU for line detection
python3 train.py rnu line_detection
```

---

## Training Workflow

### Step 1: Prepare Training Data

You need **images** and **masks** (labels) for training.

**Option A: Manual Annotation**

- Use the Mask Creator UI at `/mask-creator`
- Upload image
- Draw mask
- Save (automatically saved to `images_UNET_line_detection/` and `masks_UNET_line_detection/`)

**Option B: Generate from Web Learning**

- Extract data from websites about your domain
- Generate synthetic training pairs
- Save to training folders

**Option C: Use Pre-existing Data**

```bash
# Directory structure:
images_UNET_line_detection/
  - image_0001.png
  - image_0002.png
  - ...

masks_UNET_line_detection/
  - image_0001.png
  - image_0002.png
  - ...
```

### Step 2: Save Training Data

```bash
# Data is automatically saved when using Mask Creator UI
# Or save programmatically:

POST /save_mask?model=unet&type=line_detection
Files:
  - original: image.jpg
  - mask: mask.png

# This saves to:
# - images_UNET_line_detection/image_0001.png
# - masks_UNET_line_detection/image_0001.png
```

### Step 3: Start Training

```bash
# Via API
POST /train?model=unet&type=line_detection

# Via CLI
python3 train.py unet line_detection
```

### Step 4: Monitor Training

```bash
# Check training progress
GET /train/status?model=unet&type=line_detection

# Returns latest logs and current loss
```

### Step 5: Use Trained Model

Model is automatically saved as:

```
model_UNET_LINE_DETECTION.pth
```

Make predictions:

```bash
POST /predict?model=unet&type=line_detection
File: image.jpg
```

---

## Integrate Web Learning with Training

### Workflow: Learn → Generate Labels → Train

```python
from web_scraper import WebScraper
from web_learner import WebLearner
from model_persistence import KnowledgeBase
import requests

# Step 1: Learn about your domain
scraper = WebScraper()
learner = WebLearner()
kb = KnowledgeBase()

# Scrape and learn
result = scraper.scrape_url("https://example.com/haircut-guide")
if result['success']:
    learning = learner.extract_learning_data(result['text'])
    kb.add_knowledge('haircut_detection', learning)

# Step 2: Use learned concepts to guide annotation
knowledge = kb.get_knowledge('haircut_detection')
key_concepts = knowledge['key_terms']
print(f"Key concepts to label: {key_concepts}")

# Step 3: Prepare training data
# - Use concepts to guide manual annotation
# - Or generate synthetic training pairs
# - Save via /save_mask endpoint

# Step 4: Train model
response = requests.post("http://localhost:8000/train",
    params={"model": "unet", "type": "haircut"})
print(response.json())

# Step 5: Monitor
import time
while True:
    status = requests.get("http://localhost:8000/train/status",
        params={"model": "unet", "type": "haircut"}).json()
    print(f"Status: {status['status']}")
    if status['status'] in ['completed', 'error']:
        break
    time.sleep(5)

# Step 6: Use trained model
test_image = open('test.jpg', 'rb')
prediction = requests.post("http://localhost:8000/predict?model=unet&type=haircut",
    files={'file': test_image}).json()
print(prediction)
```

---

## Training Models

### Model 1: UNet (Line Detection)

**Purpose**: Detect lines in images (haircut lines, parking lines, etc.)

**Train:**

```bash
python3 train.py unet line_detection
```

**Use Learned Data:**

```python
kb = KnowledgeBase()
knowledge = kb.get_knowledge('line_detection')
# Use key_terms and facts to understand what lines to detect
```

### Model 2: RNU (Recurrent Neural Network U-Net)

**Purpose**: Sequence-aware segmentation

**Train:**

```bash
python3 train.py rnu line_detection
```

### Model 3: Parking Segmenter

**Purpose**: Segment parking areas in images

**Train:**

```bash
# Uses SAM (Segment Anything Model)
python3 train.py parking segmentation
```

---

## Training Parameters

### Current Settings (in train.py):

```python
# Dataset
batch_size = 4
image_size = (256, 256)

# Training
epochs = 10
learning_rate = 1e-4
loss_function = BCELoss (Binary Cross Entropy)
optimizer = Adam

# Model
model = UNet()
device = CPU or CUDA (auto-detect)
```

### Modify Training:

Edit `train.py` to change parameters:

```python
# Change epochs
for epoch in range(50):  # Increase from 10 to 50

# Change learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Increase from 1e-4

# Change batch size
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Increase from 4
```

---

## Data Preparation Guide

### Creating Training Data

**Option 1: Using Mask Creator UI**

1. Navigate to `/mask-creator`
2. Upload an image
3. Draw mask on canvas
4. Click "Save Mask"
5. Repeat for multiple images

**Option 2: Programmatic Generation**

```python
import cv2
import numpy as np
from PIL import Image

# Load image
img = cv2.imread('original.jpg')

# Create mask (white = area to detect, black = background)
mask = np.zeros(img.shape[:2], dtype=np.uint8)
# Draw white regions for what you want to detect
cv2.circle(mask, (100, 100), 30, 255, -1)

# Save
cv2.imwrite('mask.png', mask)

# Upload via API
import requests
with open('mask.png', 'rb') as m, open('original.jpg', 'rb') as o:
    requests.post('http://localhost:8000/save_mask?model=unet&type=haircut',
        files={'mask': m, 'original': o})
```

### Data Requirements

- **Minimum**: 10 image-mask pairs
- **Recommended**: 100+ pairs
- **Optimal**: 500+ pairs
- **Image size**: Any (resized to 256×256 for training)
- **Format**: PNG or JPG
- **Mask format**: Binary (black/white only)

### Data Quality Tips

1. **Consistency**: Keep annotation style consistent
2. **Variety**: Include different scales, angles, lighting
3. **Balance**: Mix easy and hard examples
4. **Precision**: Accurate mask boundaries improve training
5. **Diversity**: Include edge cases and variations

---

## Training Status Codes

```
- "not_started"  → Training hasn't begun yet
- "running"      → Training in progress
- "completed"    → Training finished successfully
- "error"        → Training failed with error
```

---

## Common Issues

### Issue: "No training data found"

**Cause**: No images in `images_UNET_line_detection/` folder

**Fix**:

1. Use Mask Creator UI to create training data
2. Or manually place images and masks in folders
3. Ensure folder names match: `images_UNET_{type}` and `masks_UNET_{type}`

### Issue: Out of Memory (OOM)

**Cause**: Batch size too large for your GPU

**Fix**:

```python
# Reduce batch size
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Was 4
```

### Issue: Training loss not decreasing

**Cause**: Learning rate too high or data issues

**Fix**:

```python
# Lower learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Was 1e-4

# Or increase epochs
for epoch in range(50):  # Was 10
```

### Issue: CUDA out of memory

**Solution**:

```python
# Force CPU training
device = torch.device("cpu")

# Or reduce image size
transforms.Resize((128, 128))  # Was 256

# Or reduce batch size
DataLoader(dataset, batch_size=1)
```

---

## Advanced: Custom Training Loop

```python
import torch
from torch import nn, optim
from unet import UNet
from utils import LineDetectionDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Setup
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = LineDetectionDataset("images_UNET_line_detection",
                               "masks_UNET_line_detection",
                               transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Custom training
for epoch in range(20):
    model.train()
    epoch_loss = 0

    for batch_idx, (img, mask) in enumerate(dataloader):
        img, mask = img.to(device), mask.to(device)

        # Forward pass
        pred = model(img)
        loss = criterion(pred, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Log progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}/20, Average Loss: {epoch_loss/len(dataloader):.4f}")

# Save
torch.save(model.state_dict(), "model_UNET_line_detection.pth")
```

---

## Monitor Training with Tensorboard

```bash
# Install tensorboard
pip install tensorboard

# In train.py, add:
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# Log loss
writer.add_scalar('Loss/train', epoch_loss, epoch)

# Start tensorboard
tensorboard --logdir=runs
# Open http://localhost:6006
```

---

## Production Best Practices

1. **Version Control Models**

   - Save with timestamp: `model_UNET_LINE_DETECTION_2024-11-08.pth`
   - Keep backup of best model

2. **Validate Regularly**

   - Test on separate validation set
   - Track accuracy metrics

3. **Monitor Performance**

   - Track training loss over time
   - Watch for overfitting

4. **Save Checkpoints**

   - Save every N epochs
   - Keep best model

5. **Document Training**
   - Record hyperparameters
   - Note dataset size and quality
   - Save training logs

---

## Next Steps

1. ✅ Prepare training data (images + masks)
2. ✅ Save via Mask Creator or API
3. ✅ Start training: `POST /train`
4. ✅ Monitor: `GET /train/status`
5. ✅ Use model: `POST /predict`
6. ✅ Iterate and improve

---

Happy training! 🚀
