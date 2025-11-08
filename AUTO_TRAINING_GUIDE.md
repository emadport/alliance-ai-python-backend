# Automated Training Pipeline

Train your AI models **completely automatically** by just providing a topic!

## 🚀 Quick Start

### One-Command Training

```bash
# The system will:
# 1. Search for websites about the topic
# 2. Scrape and learn from them
# 3. Generate synthetic training data
# 4. Automatically train your model

POST /auto-train
{
  "topic": "haircut detection",
  "model": "unet",
  "detection_type": "haircut",
  "synthetic_images": 20
}
```

## ✨ What It Does Automatically

```
Topic Input → Search Web → Scrape → Learn → Generate Data → Train → Save Model
```

### Step-by-Step Process:

1. **Search Web** 🔍

   - Searches for related Wikipedia articles, courses, tutorials
   - Generates relevant search URLs

2. **Scrape & Learn** 📚

   - Scrapes websites for content
   - Extracts entities, keywords, definitions
   - Saves knowledge to knowledge base

3. **Generate Training Data** 🎨

   - Creates synthetic training images based on learned concepts
   - Generates corresponding masks
   - Saves to proper training folders

4. **Train Model** 🤖

   - Runs training script automatically
   - Uses generated data
   - Saves trained model

5. **Save Results** 💾
   - Saves trained model
   - Stores pipeline log to MongoDB
   - Records what was learned

---

## API Endpoints

### 1. Full Auto-Training Pipeline

```bash
POST /auto-train
Content-Type: application/json

{
  "topic": "haircut detection",
  "model": "unet",
  "detection_type": "haircut",
  "urls": null,
  "synthetic_images": 20,
  "epochs": 10
}
```

**Response:**

```json
{
  "success": true,
  "topic": "haircut detection",
  "duration_seconds": 120.5,
  "steps": {
    "learning": {
      "success": true,
      "urls_processed": 5,
      "keywords_count": 150
    },
    "generation": {
      "success": true,
      "images_generated": 20
    },
    "training": {
      "success": true,
      "model_file": "model_UNET_HAIRCUT.pth"
    }
  }
}
```

### 2. Learn Only (No Training)

Just scrape and learn from web without training:

```bash
POST /auto-train/learn-only
{
  "topic": "parking lot detection",
  "urls": null
}
```

### 3. Generate Synthetic Data Only

Generate training data for a known topic:

```bash
POST /auto-train/generate-only
{
  "topic": "parking lot detection",
  "model": "unet",
  "detection_type": "parking",
  "synthetic_images": 50
}
```

### 4. Get Training History

```bash
GET /auto-train/history?limit=10
```

---

## Usage Examples

### Example 1: Train Model for Haircut Detection

```bash
curl -X POST "http://localhost:8000/auto-train" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "haircut line detection",
    "model": "unet",
    "detection_type": "haircut",
    "synthetic_images": 50,
    "epochs": 20
  }'
```

**What happens:**

1. Searches for "haircut line detection" on Wikipedia, Kaggle, etc.
2. Scrapes and learns about haircuts
3. Generates 50 synthetic images with haircut-like lines
4. Trains UNet model for 20 epochs
5. Saves as `model_UNET_HAIRCUT.pth`

### Example 2: Train for Parking Detection

```bash
curl -X POST "http://localhost:8000/auto-train" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "parking area segmentation",
    "model": "unet",
    "detection_type": "parking",
    "synthetic_images": 100,
    "epochs": 15
  }'
```

### Example 3: Use Custom URLs

```bash
curl -X POST "http://localhost:8000/auto-train" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "my custom topic",
    "model": "unet",
    "detection_type": "custom",
    "urls": [
      "https://source1.com/article",
      "https://source2.com/guide"
    ],
    "synthetic_images": 25
  }'
```

---

## Parameters Explained

| Parameter          | Type   | Default          | Description                                     |
| ------------------ | ------ | ---------------- | ----------------------------------------------- |
| `topic`            | string | Required         | What to train on (e.g., "haircut detection")    |
| `model`            | string | "unet"           | Model type (unet, rnu, etc.)                    |
| `detection_type`   | string | "line_detection" | Type of detection task                          |
| `urls`             | array  | auto-search      | Custom URLs to scrape (optional)                |
| `synthetic_images` | int    | 20               | Number of synthetic training images to generate |
| `epochs`           | int    | 10               | Number of training epochs                       |

---

## How It Works Behind the Scenes

### Step 1: Scraping

```python
# Generates URLs for topic
urls = [
  "https://en.wikipedia.org/wiki/Haircut",
  "https://www.coursera.org/search?q=haircut",
  ...
]

# Scrapes each URL
for url in urls:
    html = requests.get(url)
    text = extract_text(html)  # Clean text extraction
```

### Step 2: Learning

```python
# Extract entities, keywords, concepts
entities = extract_entities(text)
keywords = extract_keywords(text)
definitions = extract_definitions(text)

# Save to knowledge base
knowledge = {
  'key_terms': ['haircut', 'line', 'boundary', ...],
  'entities': ['Barber', 'Hair', 'Scissors', ...],
  'definitions': ['A haircut line is...', ...]
}
```

### Step 3: Generate Synthetic Data

```python
# Use learned concepts to guide generation
for i in range(synthetic_images):
    # Generate random image with shapes
    image = create_random_image()

    # Create mask with white regions (what to detect)
    mask = create_mask_with_lines()

    # Save pair
    save_image(f"images_UNET_HAIRCUT/image_{i}.png", image)
    save_image(f"masks_UNET_HAIRCUT/image_{i}.png", mask)
```

### Step 4: Train

```python
# Run standard training script
subprocess.run([
    "python", "train.py", "unet", "haircut"
])

# Model saved as: model_UNET_HAIRCUT.pth
```

---

## Real-World Examples

### Scenario 1: Train Line Detector

You want to detect lines in images (haircut lines, parking lines, etc.)

```bash
POST /auto-train
{
  "topic": "line detection",
  "detection_type": "line_detection",
  "synthetic_images": 100
}
```

Result: Model learns what lines look like from web sources, generates synthetic line images, trains automatically!

### Scenario 2: Train Specific Feature Detector

You want to detect parking spaces

```bash
POST /auto-train
{
  "topic": "parking space detection and segmentation",
  "detection_type": "parking",
  "synthetic_images": 200,
  "epochs": 20
}
```

Result: Model learns parking concepts, generates synthetic parking lot images, trains!

### Scenario 3: Multi-Domain Training

Want to train on multiple topics?

```bash
# First topic
POST /auto-train { "topic": "haircut detection" }

# Later, another topic
POST /auto-train { "topic": "parking detection" }

# Check history
GET /auto-train/history
```

---

## Monitoring Training

### Option 1: Check Status

```bash
# Check recent training jobs
GET /auto-train/history?limit=5
```

### Option 2: Custom URL Training

If the auto-search doesn't find good sources, provide your own:

```bash
POST /auto-train
{
  "topic": "specialized topic",
  "urls": [
    "https://custom-source-1.com",
    "https://custom-source-2.com"
  ]
}
```

---

## Performance Expectations

| Stage                    | Time            |
| ------------------------ | --------------- |
| Search & Scrape (5 URLs) | 30-60s          |
| Learning & Analysis      | 10-20s          |
| Generate Synthetic Data  | 5-15s           |
| Train Model (10 epochs)  | 60-180s         |
| **Total**                | **2-5 minutes** |

---

## Customization

### Adjust Synthetic Image Quality

More images = better training but slower:

```bash
POST /auto-train
{
  "topic": "detection",
  "synthetic_images": 500  # More images = better model
}
```

### Control Training Duration

More epochs = longer training but potentially better model:

```bash
POST /auto-train
{
  "topic": "detection",
  "epochs": 50  # More epochs for better convergence
}
```

### Use Specific URLs

Instead of auto-search:

```bash
POST /auto-train
{
  "topic": "my detection",
  "urls": [
    "https://expert-source-1.com",
    "https://expert-source-2.com",
    "https://expert-source-3.com"
  ]
}
```

---

## Troubleshooting

### Issue: Network errors during scraping

**Fix**: Provide custom URLs that work:

```bash
POST /auto-train
{
  "topic": "my topic",
  "urls": ["https://working-url.com"]
}
```

### Issue: Poor synthetic data quality

**Fix**: Increase sample diversity or provide better source URLs

### Issue: Training takes too long

**Fix**: Reduce epochs or synthetic images:

```bash
{
  "synthetic_images": 10,
  "epochs": 5
}
```

---

## Advanced: Programmatic Usage

```python
from auto_trainer import AutoTrainer

trainer = AutoTrainer()

# Run complete pipeline
result = trainer.auto_train_end_to_end(
    topic="haircut detection",
    model="unet",
    detection_type="haircut",
    synthetic_images=50,
    epochs=15
)

print(f"Training complete!")
print(f"Model saved: {result['steps']['training']['model_file']}")
print(f"Duration: {result['duration_seconds']:.1f}s")
```

---

## What Gets Saved

After training, you have:

```
1. Model File
   → model_UNET_HAIRCUT.pth

2. Knowledge Base
   → knowledge_base/haircut_detection.json

3. Training Data
   → images_UNET_HAIRCUT/
   → masks_UNET_HAIRCUT/

4. Pipeline Log
   → MongoDB (auto_training collection)
```

---

## Next Steps

1. ✅ Start backend: `python3 -m uvicorn server:app --reload`
2. ✅ Make auto-train request with a topic
3. ✅ Watch the pipeline run automatically
4. ✅ Use the trained model!

---

**Happy automated training!** 🎉
