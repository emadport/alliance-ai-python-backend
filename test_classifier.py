#!/usr/bin/env python3
"""
Test script for the classifier module
Run this to verify the classification system works
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def test_classifier_import():
    """Test that classifier can be imported"""
    try:
        from classifier import get_predictor, PredictionModel
        print("✓ Classifier module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import classifier: {e}")
        return False


def test_predictor_initialization():
    """Test that predictor can be initialized"""
    try:
        from classifier import get_predictor
        predictor = get_predictor()
        print("✓ Predictor initialized successfully")
        print(f"  - Device: {predictor.device}")
        print(f"  - Classifier: {type(predictor.classifier).__name__}")
        print(f"  - Regressor: {type(predictor.regressor).__name__}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize predictor: {e}")
        return False


def test_image_classification():
    """Test image classification"""
    try:
        from PIL import Image
        import numpy as np
        from io import BytesIO
        from classifier import get_predictor
        
        # Create a simple test image
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Convert to bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        img_content = img_bytes.read()
        
        # Classify
        predictor = get_predictor()
        result = predictor.classify_image(img_content)
        
        print("✓ Image classification works")
        print(f"  - Prediction: {result['prediction']}")
        print(f"  - Confidence: {result['confidence']:.2%}")
        return True
    except Exception as e:
        print(f"✗ Image classification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_classification():
    """Test CSV classification"""
    try:
        from io import BytesIO
        from classifier import get_predictor
        
        # Create test CSV
        csv_content = b"value1,value2,value3\n0.5,0.6,0.7\n0.8,0.9,1.0\n"
        
        # Classify
        predictor = get_predictor()
        result = predictor.classify_csv(csv_content)
        
        print("✓ CSV classification works")
        print(f"  - Prediction: {result['prediction']}")
        print(f"  - Confidence: {result['confidence']:.2%}")
        print(f"  - Data Summary: {result['data_summary']}")
        return True
    except Exception as e:
        print(f"✗ CSV classification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_classification():
    """Test JSON classification"""
    try:
        from classifier import get_predictor
        
        # Create test JSON
        json_content = b'{"data": [0.5, 0.6, 0.7], "metrics": {"avg": 0.6}}'
        
        # Classify
        predictor = get_predictor()
        result = predictor.classify_json(json_content)
        
        print("✓ JSON classification works")
        print(f"  - Prediction: {result['prediction']}")
        print(f"  - Confidence: {result['confidence']:.2%}")
        print(f"  - Data Summary: {result['data_summary']}")
        return True
    except Exception as e:
        print(f"✗ JSON classification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Classification System Test Suite")
    print("="*60 + "\n")
    
    tests = [
        ("Import Classifier Module", test_classifier_import),
        ("Initialize Predictor", test_predictor_initialization),
        ("Image Classification", test_image_classification),
        ("CSV Classification", test_csv_classification),
        ("JSON Classification", test_json_classification),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n[Testing] {test_name}")
        print("-" * 60)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status:10} - {test_name}")
    
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

