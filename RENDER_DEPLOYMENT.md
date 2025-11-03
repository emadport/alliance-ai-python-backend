# Render.com Deployment Guide

## Overview

This backend has two deployment modes:

### 1. **Production Mode (Lightweight)** ✅ Recommended for Free Tier
- Uses `requirements-production.txt`
- Memory usage: ~400MB (fits in 512MB free tier)
- Features:
  - ✅ UNet line/camera detection
  - ✅ Mask saving and training
  - ✅ Model predictions
  - ❌ Parking segmentation (SAM model disabled)

### 2. **Full Features Mode** 💰 Requires Paid Plan
- Uses `requirements-dev.txt` or `requirements.txt`
- Memory usage: ~3.5GB (requires 4GB+ RAM plan)
- Features:
  - ✅ All production features
  - ✅ Parking segmentation with SAM model
  - ✅ Gradio interface

---

## Deployment Steps for Render.com Free Tier

### Step 1: Configure Environment
The `render.yaml` is already configured for production mode.

### Step 2: Deploy
1. Push your code to GitHub
2. Connect your repo to Render.com
3. Render will automatically detect `render.yaml`
4. Deploy!

### Step 3: Set Environment Variables
In Render dashboard, verify these are set:
- `PORT` = 10000 (auto-set)
- `MONGO_URI` = your MongoDB connection string
- `ENABLE_PARKING_SEGMENTER` = false (for free tier)

---

## Troubleshooting

### Out of Memory Error
**Symptom:** "Out of memory (used over 512Mi)"

**Solution:**
- Ensure `ENABLE_PARKING_SEGMENTER=false`
- Verify Dockerfile isn't downloading SAM model
- Use `requirements-production.txt`

### Port Binding Error
**Symptom:** "No open ports detected"

**Solution:**
- Wait 2-3 minutes for health check
- Verify uvicorn is binding to `0.0.0.0:$PORT`
- Check server logs for startup errors

---

## Enabling Parking Segmentation (Paid Plans Only)

If you upgrade to a plan with 4GB+ RAM:

1. Set environment variable: `ENABLE_PARKING_SEGMENTER=true`
2. Update Dockerfile to download SAM:
   ```dockerfile
   RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h.pth
   ```
3. Use `requirements.txt` or `requirements-dev.txt`
4. Redeploy

---

## Local Development

For local development with all features:

```bash
# Install all dependencies including SAM
pip install -r requirements-dev.txt

# Enable parking segmentation
export ENABLE_PARKING_SEGMENTER=true

# Run server
python server.py
```

---

## Memory Usage Breakdown

| Component | RAM Usage |
|-----------|-----------|
| FastAPI + Uvicorn | ~50MB |
| PyTorch + Torchvision | ~200MB |
| UNet Model | ~50MB |
| OpenCV + PIL | ~50MB |
| **Total (Production)** | **~350MB** ✅ |
| | |
| SAM Model (vit_h) | +2.5GB |
| Segment Anything | +500MB |
| **Total (Full Features)** | **~3.5GB** 💰 |

---

## Cost Comparison

| Plan | RAM | Cost | Parking Segmentation |
|------|-----|------|---------------------|
| Free | 512MB | $0/mo | ❌ No |
| Starter | 512MB | $7/mo | ❌ No |
| Standard | 2GB | $25/mo | ❌ No (needs 3.5GB) |
| Pro | 4GB | $85/mo | ✅ Yes |

---

## Questions?

- Deployment issue? Check Render logs
- Feature not working? Verify environment variables
- Need parking segmentation? Upgrade to 4GB+ plan

