"""
Parking Area Segmentation using Segment Anything Model (SAM)
"""
import cv2
import numpy as np
from typing import List, Dict
import io
import base64
from PIL import Image
from huggingface_hub import hf_hub_download


try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment_anything not installed. Parking segmentation will not work.")

class ParkingSegmenter:
    def __init__(self, checkpoint_path: str = "sam_vit_h.pth"):
        """
        Initialize the parking segmenter with SAM model
        
        Args:
            checkpoint_path: Path to SAM checkpoint file
        """
        if not SAM_AVAILABLE:
            raise ImportError("segment_anything package is required for parking segmentation")
        
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """Load the SAM model"""
        try:
            # Download checkpoint from Hugging Face if not already cached
            self.checkpoint_path = hf_hub_download(
                repo_id="Emad-askari/alliance-ai-models",
                filename="sam_vit_h.pth"
            )
            
            # Load SAM model using the downloaded checkpoint
            self.model = sam_model_registry["vit_h"](checkpoint=self.checkpoint_path)
            self.predictor = SamPredictor(self.model)
            print("SAM model loaded successfully")
            
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            print("Make sure the checkpoint file is accessible")
            raise
    
    def segment_parking_areas(self, image: np.ndarray) -> Dict:
        """
        Segment parking areas from an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing masks, polygons, and visualization
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Use automatic mask generation for whole image segmentation
        from segment_anything import SamAutomaticMaskGenerator
        
        # Configure generator to produce fewer, larger segments (parking spaces)
        # min_mask_region_area: minimum area in pixels for a mask (higher = fewer segments)
        # points_per_side: density of sampling points (lower = fewer segments)
        # pred_iou_thresh: minimum IOU threshold for quality filtering
        mask_generator = SamAutomaticMaskGenerator(
            self.model,
            min_mask_region_area=5000,  # Only keep masks with area > 5000 pixels
            points_per_side=32,  # Fewer points = fewer segments (default is 64)
            pred_iou_thresh=0.88,  # Higher threshold = better quality masks
        )
        anns = mask_generator.generate(image)
        
        # Convert annotations to polygons
        polygons = self._annotations_to_polygons(anns)
        
        # Create visualization from polygons (not raw annotations) to match count
        visualization = self._create_visualization_from_polygons(image, polygons)
        
        return {
            "masks": None,  # Store annotations instead
            "scores": [ann["predicted_iou"] for ann in anns],
            "polygons": polygons,
            "visualization": visualization
        }
    
    def _masks_to_polygons(self, masks: np.ndarray) -> List[Dict]:
        """
        Convert binary masks to polygon coordinates
        
        Args:
            masks: Array of binary masks
            
        Returns:
            List of polygon dictionaries
        """
        polygons = []
        
        for i in range(masks.shape[0]):
            mask = masks[i].astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                # Simplify contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 3:  # At least 3 points for a polygon
                    # Convert to list of [x, y] coordinates
                    points = approx.reshape(-1, 2).tolist()
                    
                    polygons.append({
                        "points": points,
                        "area": cv2.contourArea(contour)
                    })
        
        return polygons
    
    def _annotations_to_polygons(self, anns: List[Dict]) -> List[Dict]:
        """
        Convert SAM annotations to polygon coordinates
        
        Args:
            anns: List of annotation dictionaries from SAM
            
        Returns:
            List of polygon dictionaries
        """
        polygons = []
        
        for ann in anns:
            # Get segmentation mask
            mask = ann["segmentation"]
            area = ann["area"]
            
            # Convert mask to polygons
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                # Simplify contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 3:  # At least 3 points for a polygon
                    # Convert to list of [x, y] coordinates
                    points = approx.reshape(-1, 2).tolist()
                    
                    # Only include if area is reasonable (filter tiny noise)
                    # Minimum area of 1000 pixels to filter out small fragments
                    if area > 1000:  # Minimum area threshold
                        polygons.append({
                            "points": points,
                            "area": area
                        })
        
        return polygons
    
    def _create_visualization_from_anns(
        self, 
        image: np.ndarray, 
        anns: List[Dict]
    ) -> np.ndarray:
        """
        Create visualization from SAM annotations
        
        Args:
            image: Original image
            anns: List of annotations
            
        Returns:
            Visualized image
        """
        vis_image = image.copy()
        
        # Sort annotations by area (largest first)
        sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)
        
        # Draw masks with colors
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        
        for i, ann in enumerate(sorted_anns):
            if ann["area"] < 1000:  # Skip tiny masks (same threshold as polygon generation)
                continue
                
            color = colors[i % len(colors)]
            mask = ann["segmentation"]
            
            # Create colored overlay
            color_mask = np.zeros_like(vis_image)
            color_mask[mask] = color
            
            # Blend with original image
            vis_image = cv2.addWeighted(vis_image, 0.7, color_mask, 0.3, 0)
            
            # Draw contour outline
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis_image, contours, -1, color, 2)
        
        return vis_image
    
    def _create_visualization_from_polygons(
        self, 
        image: np.ndarray, 
        polygons: List[Dict]
    ) -> np.ndarray:
        """
        Create visualization of detected parking areas from polygons
        
        Args:
            image: Original image
            polygons: Polygon coordinates
            
        Returns:
            Visualized image
        """
        vis_image = image.copy()
        
        # Draw polygons on image
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        
        for i, polygon in enumerate(polygons):
            color = colors[i % len(colors)]
            points = np.array(polygon["points"], dtype=np.int32)
            
            # Draw filled polygon
            cv2.fillPoly(vis_image, [points], color + (50,))  # Semi-transparent
            
            # Draw polygon outline
            cv2.polylines(vis_image, [points], True, color, 2)
        
        return vis_image
    
    def _create_visualization(
        self, 
        image: np.ndarray, 
        masks: np.ndarray, 
        polygons: List[Dict]
    ) -> np.ndarray:
        """
        Create visualization of detected parking areas
        
        Args:
            image: Original image
            masks: Detected masks
            polygons: Polygon coordinates
            
        Returns:
            Visualized image
        """
        vis_image = image.copy()
        
        # Draw polygons on image
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        
        for i, polygon in enumerate(polygons):
            color = colors[i % len(colors)]
            points = np.array(polygon["points"], dtype=np.int32)
            
            # Draw filled polygon
            cv2.fillPoly(vis_image, [points], color + (50,))  # Semi-transparent
            
            # Draw polygon outline
            cv2.polylines(vis_image, [points], True, color, 2)
        
        return vis_image
    
    def image_to_base64(self, image: np.ndarray) -> str:
        """
        Convert numpy image to base64 string
        
        Args:
            image: Image as numpy array
            
        Returns:
            Base64 encoded image string
        """
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    def process_image_bytes(self, image_bytes: bytes) -> Dict:
        """
        Process image from bytes
        
        Args:
            image_bytes: Image as bytes
            
        Returns:
            Dictionary with results
        """
        # Convert bytes to numpy array
        image = Image.open(io.BytesIO(image_bytes))
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Segment parking areas
        results = self.segment_parking_areas(image_np)
        
        # Convert visualization to base64
        vis_base64 = self.image_to_base64(results["visualization"])
        
        return {
            "success": True,
            "image": vis_base64,
            "polygons": results["polygons"],
            "num_spaces": len(results["polygons"]),
            "message": f"Detected {len(results['polygons'])} parking spaces"
        }

