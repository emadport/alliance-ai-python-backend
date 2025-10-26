import gradio as gr
import numpy as np
from PIL import Image
import os
import glob
import requests

def get_next_image_number():
    """Get the next available image number"""
    # Check existing images in images folder
    existing = glob.glob("images/image_*.png")
    if not existing:
        return 1
    
    # Extract numbers
    numbers = []
    for f in existing:
        try:
            name = os.path.basename(f)
            num = int(name.split("_")[1].split(".")[0])
            numbers.append(num)
        except:
            pass
    
    return max(numbers) + 1 if numbers else 1

def extract_mask_and_original(editor_output):
    if editor_output is None:
        return None, None, None
    
    try:
        # ImageEditor returns a dict with layers
        if isinstance(editor_output, dict):
            # Get original background image
            background = editor_output.get("background")
            original_image = None
            if background:
                if isinstance(background, Image.Image):
                    original_image = background
                else:
                    original_image = Image.fromarray(np.array(background))
            
            # Get the mask layer (what was drawn)
            mask_layer = editor_output.get("layers")
            if mask_layer is None:
                return None, None, None
            
            # Get the first layer which is typically the drawn mask
            if isinstance(mask_layer, list) and len(mask_layer) > 0:
                mask_image = mask_layer[0]
            else:
                mask_image = mask_layer
            
            # Convert to numpy array if PIL Image
            if isinstance(mask_image, Image.Image):
                mask_array = np.array(mask_image)
            else:
                mask_array = mask_image
            
            # Convert to grayscale for mask
            if len(mask_array.shape) == 3:
                # Extract just white pixels (the drawn parts)
                # Create binary mask: white parts become 255, everything else 0
                mask_binary = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)
                # If any channel is > threshold, consider it a white pixel
                white_threshold = 200
                mask_binary[np.any(mask_array > white_threshold, axis=2)] = 255
                mask_pil = Image.fromarray(mask_binary, mode="L")
            else:
                mask_pil = Image.fromarray(mask_array, mode="L")
            
            # Get next image number
            img_num = get_next_image_number()
            filename = f"image_{img_num:04d}"
            
            # Save original image
            if original_image:
                original_path = f"images/{filename}.png"
                original_image.save(original_path)
            
            # Save mask
            mask_path = f"masks/{filename}.png"
            mask_pil.save(mask_path)
            
            return mask_pil, mask_path, f"Saved as {filename}.png"
        
        return None, None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None, f"Error: {str(e)}"

def load_image_from_url(url):
    """Load image from URL"""
    if url:
        try:
            from PIL import Image
            import requests
            from io import BytesIO
            
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            return img
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    return None

with gr.Blocks() as demo:
    gr.Markdown("# Draw Mask with Image as Guide")
    
    with gr.Tab("Upload and Draw"):
        with gr.Row():
            image_url_input = gr.Textbox(
                label="Or enter image URL",
                placeholder="data:image/... or http://..."
            )
        
        with gr.Row():
            image_input = gr.ImageEditor(
                label="Upload Image and Draw Mask",
                type="pil"
            )
        
        with gr.Row():
            mask_preview = gr.Image(label="Saved Mask Preview", type="pil")
        
        with gr.Row():
            save_btn = gr.Button("Save Image and Mask", variant="primary")
            download = gr.File(label="Download Mask")
            status = gr.Textbox(label="Status", interactive=False)
        
        def load_url_image(url):
            img = load_image_from_url(url)
            return img
        
        image_url_input.submit(
            load_url_image,
            inputs=image_url_input,
            outputs=image_input
        )
        
        save_btn.click(
            extract_mask_and_original,
            inputs=image_input,
            outputs=[mask_preview, download, status]
        )

demo.launch(share=False, server_name="0.0.0.0", server_port=7860)