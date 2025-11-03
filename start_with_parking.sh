#!/bin/bash

# Start the Python backend with parking segmentation enabled
# This requires ~3GB RAM and will download SAM model on first run

echo "🚀 Starting Alliance AI Backend with Parking Segmentation..."
echo ""
echo "⚠️  Note: This requires:"
echo "   - ~3GB RAM"
echo "   - segment-anything installed (pip install -r requirements-dev.txt)"
echo "   - First run will download 2.4GB SAM model"
echo ""

# Enable parking segmentation
export ENABLE_PARKING_SEGMENTER=true

# Start server
python3 server.py

