#!/bin/bash

# Start the Python backend in basic mode (no parking segmentation)
# This is lightweight and works with requirements-production.txt

echo "🚀 Starting Alliance AI Backend (Basic Mode)..."
echo ""
echo "✅ Features enabled:"
echo "   - UNet line/camera detection"
echo "   - Mask creation and training"
echo "   - Model predictions"
echo ""
echo "❌ Parking segmentation disabled (use start_with_parking.sh to enable)"
echo ""

# Disable parking segmentation explicitly
export ENABLE_PARKING_SEGMENTER=false

# Start server
python3 server.py

