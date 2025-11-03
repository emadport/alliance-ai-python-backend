# Quick Start Guide 🚀

## Choose Your Mode

### Option 1: Basic Mode (Recommended for most use)
**Lightweight, fast, uses ~400MB RAM**

```bash
./start_basic.sh
```

**Features:**
- ✅ UNet line detection
- ✅ Camera detection  
- ✅ Mask creation & training
- ✅ Model predictions
- ❌ Parking segmentation (disabled)

---

### Option 2: Full Features Mode (Includes Parking)
**Heavy, requires ~3GB RAM and downloads 2.4GB model**

**First time setup:**
```bash
pip install -r requirements-dev.txt
```

**Then start:**
```bash
./start_with_parking.sh
```

**Features:**
- ✅ All basic features
- ✅ Parking segmentation with SAM model

**⚠️ First run:** Will download 2.4GB SAM model (~5-10 minutes)

---

## Manual Start

### Basic Mode
```bash
export ENABLE_PARKING_SEGMENTER=false
python3 server.py
```

### With Parking
```bash
export ENABLE_PARKING_SEGMENTER=true
python3 server.py
```

---

## Verify Server is Running

```bash
curl http://localhost:8000/health
```

Should return:
```json
{"status":"ok","mongodb":"connected"}
```

---

## Troubleshooting

### "Parking segmenter not available" error
- This is expected in basic mode
- To enable: Use `./start_with_parking.sh`
- Make sure you installed: `pip install -r requirements-dev.txt`

### "Out of memory" or system slow
- Close other applications
- SAM model requires ~3GB RAM
- Consider using basic mode instead

### Port already in use
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

---

## File Structure

```
requirements-production.txt  → Lightweight (for Render.com free tier)
requirements-dev.txt        → Full features (for local development)
requirements.txt           → All packages (backward compatibility)

start_basic.sh            → Start without parking (fast)
start_with_parking.sh     → Start with parking (heavy)

server.py                 → Main server file
```

---

## Deployment to Render.com

Automatically uses **production mode** (no parking):
- Configured in `render.yaml`
- Uses `requirements-production.txt`
- Fits in 512MB free tier ✅

See `RENDER_DEPLOYMENT.md` for details.

