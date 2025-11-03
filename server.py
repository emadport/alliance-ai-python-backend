from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import io
import base64
import os
from datetime import datetime
from unet import UNet
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

# Try to import parking segmenter
try:
    from parking_segmenter import ParkingSegmenter
    PARKING_AVAILABLE = True
except ImportError:
    PARKING_AVAILABLE = False
    print("Warning: Parking segmenter not available")

app = FastAPI()

# MongoDB connection - lazy init
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://emadaskari_db_user:kLnkczzLG9QpgXn6@cluster0.mongodb.net/emadaskari_db?retryWrites=true&w=majority")
mongo_client = None
db = None

def get_db():
    global mongo_client, db
    if db is None:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client.emadaskari_db
    return db

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

# Try to load trained weights if available
try:
    model.load_state_dict(torch.load("model_UNET_line_detection.pth", map_location=device))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"No pre-trained model found or error loading: {e}")

# Initialize parking segmenter (only if explicitly enabled)
# SAM model is ~2.5GB and requires significant RAM
# Set ENABLE_PARKING_SEGMENTER=true in environment to enable
parking_segmenter = None
ENABLE_PARKING = os.getenv("ENABLE_PARKING_SEGMENTER", "false").lower() == "true"

if PARKING_AVAILABLE and ENABLE_PARKING:
    try:
        print("Initializing parking segmenter (this may take a while and requires ~3GB RAM)...")
        parking_segmenter = ParkingSegmenter(checkpoint_path="sam_vit_h.pth")
        print("Parking segmenter initialized successfully")
    except Exception as e:
        print(f"Could not initialize parking segmenter: {e}")
        parking_segmenter = None
else:
    if not PARKING_AVAILABLE:
        print("Parking segmenter not available (segment_anything not installed)")
    elif not ENABLE_PARKING:
        print("Parking segmenter disabled (set ENABLE_PARKING_SEGMENTER=true to enable)")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def preprocess_image(image_bytes):
    """Convert image bytes to tensor"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

def predict_mask(image_tensor):
    """Get prediction from model"""
    with torch.no_grad():
        pred = model(image_tensor.to(device))
    return pred

def mask_to_base64(pred_mask, orig_image):
    """Convert mask to image with overlay"""
    mask = pred_mask.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Convert PIL image to OpenCV format
    orig_cv = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)
    orig_resized = cv2.resize(orig_cv, (256, 256))
    
    # Find contours and draw
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = orig_resized.copy()
    
    for cnt in contours:
        if len(cnt) > 10:
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
    
    # Convert back to base64
    _, buffer = cv2.imencode('.jpg', result)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64

@app.get("/")
def root():
    return {"message": "Alliance AI Python Backend", "status": "running"}

@app.get("/health")
def health_check():
    try:
        get_db()
        return {"status": "ok", "mongodb": "connected"}
    except:
        return {"status": "ok", "mongodb": "not connected"}

@app.get("/datasets")
def get_datasets():
    """Get all saved datasets"""
    try:
        datasets = list(get_db().datasets.find({}, {"_id": 0}).sort("created_at", -1))
        return {"success": True, "datasets": datasets}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/train")
async def train_model(request: Request):
    """Train the model"""
    try:
        import subprocess
        import os
        
        # Get model and detection type from query parameters
        model_name = request.query_params.get("model", "unet")
        detection_type = request.query_params.get("type", "line_detection")
        
        # Change to the correct directory
        os.chdir(os.path.dirname(__file__))
        
        # Create log file for training progress
        log_file = f"training_{model_name}_{detection_type}.log"
        
        # Run training with both model and detection type, redirect output to log file
        # Use -u flag for unbuffered output so logs appear immediately
        subprocess.Popen(
            ["python3", "-u", "train.py", model_name, detection_type],
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT
        )
        
        return {
            "success": True,
            "message": f"Training started for {model_name.upper()} model ({detection_type} type) in background",
            "model": model_name,
            "type": detection_type
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/train/status")
async def get_training_status(request: Request):
    """Get training progress from log file"""
    try:
        import os
        
        # Get model and detection type from query parameters
        model_name = request.query_params.get("model", "unet")
        detection_type = request.query_params.get("type", "line_detection")
        
        log_file = f"training_{model_name}_{detection_type}.log"
        
        if not os.path.exists(log_file):
            return {
                "success": True,
                "status": "not_started",
                "logs": []
            }
        
        # Read last 50 lines of log file
        with open(log_file, "r") as f:
            lines = f.readlines()
            recent_lines = lines[-50:] if len(lines) > 50 else lines
            logs = [line.strip() for line in recent_lines if line.strip()]
        
        # Determine status based on log content
        status = "running"
        if logs:
            last_line = logs[-1]
            if "Training complete" in last_line or "complete" in last_line.lower():
                status = "completed"
                
                # Delete training images and masks after successful training
                import shutil
                import glob
                
                img_folder = f"images_{model_name.upper()}_{detection_type}"
                mask_folder = f"masks_{model_name.upper()}_{detection_type}"
                
                # Delete image folder
                if os.path.exists(img_folder):
                    shutil.rmtree(img_folder)
                    print(f"Deleted training images: {img_folder}")
                
                # Delete mask folder
                if os.path.exists(mask_folder):
                    shutil.rmtree(mask_folder)
                    print(f"Deleted training masks: {mask_folder}")
                
                # Also delete the log file
                if os.path.exists(log_file):
                    os.remove(log_file)
                    print(f"Deleted training log: {log_file}")
                    
            elif "error" in last_line.lower() or "failed" in last_line.lower():
                status = "error"
        
        return {
            "success": True,
            "status": status,
            "logs": logs
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/save_mask")
async def save_mask(request: Request, original: UploadFile = File(None), mask: UploadFile = File(...)):
    """Save mask and original image to disk with compression"""
    try:
        import os
        import glob
        
        # Get model and detection type from query parameters
        model_name = request.query_params.get("model", "unet")
        detection_type = request.query_params.get("type", "line_detection")
        
        # Determine folders based on model and detection type
        # Format: images_UNET_haircut, masks_UNET_haircut
        img_folder = f"images_{model_name.upper()}_{detection_type}"
        mask_folder = f"masks_{model_name.upper()}_{detection_type}"
        
        # Create folders if they don't exist
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)
        
        # Get next image number
        existing = glob.glob(f"{img_folder}/image_*.png")
        numbers = []
        for f in existing:
            try:
                name = os.path.basename(f)
                num = int(name.split("_")[1].split(".")[0])
                numbers.append(num)
            except:
                pass
        img_num = max(numbers) + 1 if numbers else 1
        filename = f"image_{img_num:04d}"
        
        # Training image size (consistent size for model)
        TARGET_SIZE = (256, 256)
        
        # Save original image if provided with compression
        if original:
            orig_contents = await original.read()
            orig_image = Image.open(io.BytesIO(orig_contents))
            
            # Fix EXIF orientation (critical for phone photos!)
            try:
                from PIL import ImageOps
                orig_image = ImageOps.exif_transpose(orig_image)
            except Exception as e:
                print(f"Could not fix EXIF orientation: {e}")
            
            # Resize to target size for training
            orig_image = orig_image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed (remove alpha channel)
            if orig_image.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', orig_image.size, (255, 255, 255))
                if orig_image.mode == 'P':
                    orig_image = orig_image.convert('RGBA')
                rgb_image.paste(orig_image, mask=orig_image.split()[-1] if orig_image.mode in ('RGBA', 'LA') else None)
                orig_image = rgb_image
            elif orig_image.mode != 'RGB':
                # Handle any other non-RGB modes
                orig_image = orig_image.convert('RGB')
            
            # Save as optimized PNG
            orig_image.save(f"{img_folder}/{filename}.png", "PNG", optimize=True)
        
        # Save mask with compression (binary black and white only)
        mask_contents = await mask.read()
        mask_image = Image.open(io.BytesIO(mask_contents))
        
        # Resize mask to same target size
        mask_image = mask_image.resize(TARGET_SIZE, Image.Resampling.NEAREST)  # NEAREST for binary masks
        
        # Convert to binary (black and white only) for smaller file size
        mask_image = mask_image.convert('L')  # Convert to grayscale
        mask_array = np.array(mask_image)
        # Threshold to pure black (0) and white (255)
        mask_array = ((mask_array > 127) * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_array, mode='L')
        
        # Save as optimized PNG
        mask_image.save(f"{mask_folder}/{filename}.png", "PNG", optimize=True)
        
        # Save to MongoDB
        try:
            dataset_doc = {
                "model": model_name.upper(),
                "detection_type": detection_type,
                "image_filename": f"{filename}.png",
                "created_at": datetime.now(),
                "folders": {"images": img_folder, "masks": mask_folder}
            }
            get_db().datasets.insert_one(dataset_doc)
        except Exception as e:
            print(f"MongoDB error: {e}")
        
        return {
            "success": True,
            "filename": f"{filename}.png",
            "model": model_name,
            "type": detection_type,
            "folders": {"images": img_folder, "masks": mask_folder},
            "image_size": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
            "message": f"Mask and image saved (resized to {TARGET_SIZE[0]}x{TARGET_SIZE[1]} and optimized)"
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in save_mask: {error_details}")
        
        # Provide helpful error messages
        error_msg = str(e)
        if "cannot identify image file" in error_msg.lower():
            error_msg = "Unsupported image format. Please use JPG, PNG, or convert HEIC to JPG first."
        elif "decoder" in error_msg.lower():
            error_msg = "Image format error. Try converting your photo to JPG before uploading."
        elif "memory" in error_msg.lower():
            error_msg = "Image too large. Please use a smaller photo."
        
        return {
            "success": False,
            "error": error_msg
        }

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """Predict on uploaded image"""
    try:
        # Get model and detection type from query parameters
        model_name = request.query_params.get("model", "unet")
        detection_type = request.query_params.get("type", "line_detection")
        
        # Load the correct model
        model_filename = f"model_{model_name.upper()}_{detection_type}.pth"
        if os.path.exists(model_filename):
            print(f"Loading model: {model_filename}")
            model.load_state_dict(torch.load(model_filename, map_location=device))
            model.eval()
        else:
            print(f"Model {model_filename} not found, using current model")
        
        # Read image
        contents = await file.read()
        image_tensor, orig_image = preprocess_image(contents)
        
        # Predict
        pred_mask = predict_mask(image_tensor)
        
        # Create output image
        result_base64 = mask_to_base64(pred_mask, orig_image)
        
        return {
            "success": True,
            "image": result_base64,
            "model": model_name,
            "type": detection_type,
            "message": "Prediction completed"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/parking/segment")
async def segment_parking(file: UploadFile = File(...)):
    """Segment parking areas using SAM"""
    try:
        if not parking_segmenter:
            error_msg = "Parking segmenter not available. "
            if not PARKING_AVAILABLE:
                error_msg += "The segment_anything package is not installed. "
            elif not ENABLE_PARKING:
                error_msg += "The feature is disabled on this server (requires 3GB+ RAM). "
            error_msg += "This feature requires significant resources and is disabled on free hosting tiers."
            
            return {
                "success": False,
                "error": error_msg
            }
        
        # Read image
        contents = await file.read()
        
        # Fix EXIF orientation for phone photos before processing
        try:
            from PIL import ImageOps
            image = Image.open(io.BytesIO(contents))
            image = ImageOps.exif_transpose(image)
            
            # Convert back to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=95)
            contents = img_byte_arr.getvalue()
        except Exception as e:
            print(f"Could not fix EXIF orientation for parking: {e}")
            # Continue with original image
        
        # Process with parking segmenter
        result = parking_segmenter.process_image_bytes(contents)
        
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

