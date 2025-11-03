# Starting Local Development Server

## Quick Start (Basic Features)

```bash
cd /Users/emadaskari/Documents/personal/allianceai-backend

# Install production dependencies (lightweight)
pip install -r requirements-production.txt

# Start server
python server.py
```

Server will be available at: `http://localhost:8000`

**Features available:**
- ✅ UNet line/camera detection  
- ✅ Mask creation and training
- ✅ Model predictions
- ❌ Parking segmentation (disabled)

---

## Full Features (Including Parking Segmentation)

If you want to use parking segmentation locally:

### Step 1: Install full dependencies
```bash
pip install -r requirements-dev.txt
# This installs segment-anything and other heavy packages
```

### Step 2: Enable parking segmentation
```bash
# On macOS/Linux:
export ENABLE_PARKING_SEGMENTER=true

# On Windows (PowerShell):
$env:ENABLE_PARKING_SEGMENTER="true"

# On Windows (CMD):
set ENABLE_PARKING_SEGMENTER=true
```

### Step 3: Start server
```bash
python server.py
```

**First time startup:**
- SAM model (~2.4GB) will auto-download from Hugging Face
- This may take 5-10 minutes depending on your internet speed
- Requires ~3GB RAM to run

**Features available:**
- ✅ All basic features
- ✅ Parking segmentation with SAM model

---

## Troubleshooting

### "Parking segmenter disabled" error
**Solution:** Set environment variable before starting:
```bash
export ENABLE_PARKING_SEGMENTER=true
python server.py
```

### Out of Memory
**Symptom:** Python crashes or system becomes slow

**Solution:** Close other applications. SAM requires ~3GB RAM

### SAM model download fails
**Symptom:** "Could not initialize parking segmenter"

**Solution:** 
1. Check internet connection
2. Manually download: https://huggingface.co/Emad-askari/alliance-ai-models
3. Place `sam_vit_h.pth` in backend directory

---

## Quick Commands Reference

### Basic Development (No Parking)
```bash
cd /Users/emadaskari/Documents/personal/allianceai-backend
python server.py
```

### Full Features (With Parking)
```bash
cd /Users/emadaskari/Documents/personal/allianceai-backend
export ENABLE_PARKING_SEGMENTER=true
python server.py
```

### Check if server is running
```bash
curl http://localhost:8000/health
```

### Test parking endpoint
```bash
curl http://localhost:8000/parking/segment
```

