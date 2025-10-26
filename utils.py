import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class HaircutDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([
            fname for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Assume mask has the same base name but .png extension
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, base_name + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def draw_haircut_line(pred_mask, original_image):
    """
    pred_mask: tensor or numpy array of shape (1, H, W) or (H, W)
    original_image: OpenCV image (BGR format)
    """
    # Convert tensor to numpy if needed
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().squeeze().numpy()
    else:
        pred_mask = np.squeeze(pred_mask)

    # Threshold
    mask = (pred_mask > 0.1).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        rows, cols = original_image.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        cv2.line(original_image, (cols - 1, righty), (0, lefty), (0, 255, 0), 1)
        
        cv2.drawContours(original_image, [largest], -1,(150, 100, 0), 2)


    return original_image
