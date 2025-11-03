# Parking Area Segmentation Setup

This feature uses Meta's Segment Anything Model (SAM) to automatically detect and segment parking spaces from parking lot images.

## Prerequisites

1. **Install segment-anything package**:

   ```bash
   pip install segment-anything
   ```

2. **Download SAM checkpoint**:
   You need to download the SAM ViT-H checkpoint file:

   - Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   - Save it as `sam_vit_h.pth` in the `backend-deploy` directory

   Or use wget/curl:

   ```bash
   cd backend-deploy
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h.pth
   ```

## How It Works

1. **Upload Image**: User uploads a parking lot image
2. **SAM Segmentation**: The model automatically segments all objects in the image
3. **Mask to Polygons**: Detected masks are converted to polygon coordinates
4. **Visualization**: Parking spaces are visualized with different colors

## API Endpoint

- **POST** `/parking/segment`
- **Input**: Image file (multipart/form-data)
- **Output**: JSON with base64 image and polygon data

## Notes

- SAM model file is large (~2.4GB for ViT-H)
- Make sure you have enough disk space
- First run will download model weights automatically
- GPU recommended for faster inference

## Alternative Checkpoints

If you want a smaller/faster model:

- ViT-L: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
- ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Update `checkpoint_path` in `ParkingSegmenter` initialization accordingly.
