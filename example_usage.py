#!/usr/bin/env python3
"""
Example usage of the classifier module
This shows how to use the classifier programmatically
"""

from classifier import get_predictor
from PIL import Image
import numpy as np
from io import BytesIO
import json


def example_1_image_classification():
    """Example 1: Classify an image"""
    print("\n" + "="*60)
    print("Example 1: Image Classification")
    print("="*60)
    
    # Get predictor
    predictor = get_predictor()
    
    # Create a sample image
    img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_content = img_bytes.read()
    
    # Classify
    result = predictor.classify_image(img_content)
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nAll probabilities:")
    for label, prob in result['all_probabilities'].items():
        print(f"  {label}: {prob:.4f} ({prob*100:.2f}%)")


def example_2_csv_classification():
    """Example 2: Classify CSV data"""
    print("\n" + "="*60)
    print("Example 2: CSV Classification")
    print("="*60)
    
    predictor = get_predictor()
    
    # Create sample CSV
    csv_content = b"""value1,value2,value3,value4
0.1,0.2,0.15,0.12
0.08,0.18,0.10,0.14
0.12,0.25,0.20,0.18"""
    
    result = predictor.classify_csv(csv_content)
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nData Summary:")
    for key, value in result['data_summary'].items():
        print(f"  {key}: {value}")


def example_3_json_classification():
    """Example 3: Classify JSON data"""
    print("\n" + "="*60)
    print("Example 3: JSON Classification")
    print("="*60)
    
    predictor = get_predictor()
    
    # Create sample JSON
    json_data = {
        "metrics": {
            "accuracy": 0.85,
            "precision": 0.90,
            "recall": 0.88
        },
        "timestamps": [0.5, 0.6, 0.7],
        "scores": [0.8, 0.75, 0.82]
    }
    json_content = json.dumps(json_data).encode('utf-8')
    
    result = predictor.classify_json(json_content)
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nData Summary:")
    for key, value in result['data_summary'].items():
        print(f"  {key}: {value}")


def example_4_image_regression():
    """Example 4: Regression on image"""
    print("\n" + "="*60)
    print("Example 4: Image Regression")
    print("="*60)
    
    predictor = get_predictor()
    
    # Create a sample image
    img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_content = img_bytes.read()
    
    # Regress
    result = predictor.regress_image(img_content)
    
    print(f"Predicted Value: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nImage Features:")
    for key, value in result['details'].items():
        print(f"  {key}: {value}")


def example_5_csv_regression():
    """Example 5: Regression on CSV data"""
    print("\n" + "="*60)
    print("Example 5: CSV Regression")
    print("="*60)
    
    predictor = get_predictor()
    
    # Create sample CSV with numeric features
    csv_content = b"""feature1,feature2,feature3,feature4,feature5
0.1,0.2,0.3,0.4,0.5
0.15,0.25,0.35,0.45,0.55
0.12,0.22,0.32,0.42,0.52"""
    
    result = predictor.regress_csv(csv_content)
    
    print(f"Predicted Value: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nData Analysis:")
    for key, value in result['details'].items():
        print(f"  {key}: {value}")


def example_6_batch_predictions():
    """Example 6: Batch predictions"""
    print("\n" + "="*60)
    print("Example 6: Batch Predictions")
    print("="*60)
    
    predictor = get_predictor()
    
    # Create multiple CSV samples
    samples = [
        b"value1,value2\n0.1,0.2\n0.3,0.4",
        b"value1,value2\n0.5,0.6\n0.7,0.8",
        b"value1,value2\n0.9,1.0\n0.2,0.3",
    ]
    
    print("\nClassifying 3 CSV files...")
    for i, sample in enumerate(samples, 1):
        result = predictor.classify_csv(sample)
        print(f"  Sample {i}: {result['prediction']} ({result['confidence']:.2%})")


def example_7_real_image_file():
    """Example 7: Classify a real image file"""
    print("\n" + "="*60)
    print("Example 7: Classify Real Image (if available)")
    print("="*60)
    
    import os
    from pathlib import Path
    
    # Look for any image files in the backend directory
    backend_dir = Path(__file__).parent
    image_files = list(backend_dir.glob('**/*.png')) + list(backend_dir.glob('**/*.jpg'))
    
    if not image_files:
        print("No image files found in backend directory")
        print("To use this example, add an image file to the backend folder")
        return
    
    predictor = get_predictor()
    image_path = image_files[0]
    
    print(f"\nClassifying: {image_path.name}")
    
    with open(image_path, 'rb') as f:
        img_content = f.read()
    
    result = predictor.classify_image(img_content)
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    print("\n" + "█"*60)
    print("█  CLASSIFIER MODULE - USAGE EXAMPLES")
    print("█"*60)
    
    try:
        example_1_image_classification()
        example_2_csv_classification()
        example_3_json_classification()
        example_4_image_regression()
        example_5_csv_regression()
        example_6_batch_predictions()
        example_7_real_image_file()
        
        print("\n" + "█"*60)
        print("█  ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("█"*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()

