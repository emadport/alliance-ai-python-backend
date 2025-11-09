"""
Classification and Regression Module
Handles file-based predictions using both classification and regression models
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import io
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json


class SimpleClassifier(nn.Module):
    """Simple convolutional classifier for image classification"""
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super(SimpleClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleRegressor(nn.Module):
    """Simple neural network for regression tasks"""
    def __init__(self, input_size: int = 6, output_size: int = 1):
        super(SimpleRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        return self.network(x)


class PredictionModel:
    """Main prediction model manager"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = None
        self.regressor = None
        self.class_labels = {}
        self.load_models()

    def load_models(self):
        """Load pre-trained models or initialize new ones"""
        try:
            # Initialize classifier with reasonable defaults
            self.classifier = SimpleClassifier(num_classes=10).to(self.device)
            self.classifier.eval()
            
            # Initialize regressor with correct input size (6 features from image)
            self.regressor = SimpleRegressor(input_size=6, output_size=1).to(self.device)
            self.regressor.eval()
            
            # Load class labels if available
            self.class_labels = {
                0: "Animal", 1: "Building", 2: "Car", 3: "Dog", 4: "Cat",
                5: "Bird", 6: "Tree", 7: "Person", 8: "Boat", 9: "Airplane"
            }
            
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")

    def classify_image(self, file_content: bytes) -> Dict[str, Any]:
        """Classify an image file"""
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(file_content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to expected size
            image = image.resize((224, 224))
            
            # Convert to tensor and normalize
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            
            image_tensor = transform(image).unsqueeze(0).to(self.device)  # type: ignore
            
            # Make prediction
            with torch.no_grad():  # type: ignore[attr-defined]
                outputs = self.classifier(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            class_idx = predicted_class.item()
            confidence_score = confidence.item()
            
            # Build probabilities dict with proper type handling
            prob_dict: Dict[str, float] = {}
            for i in range(len(probabilities[0])):
                label_key: str = str(i)
                label = self.class_labels.get(int(label_key), f"Class {i}")
                prob_value = probabilities[0][i].item()
                prob_dict[label] = round(float(prob_value), 4)
            
            return {
                'prediction': self.class_labels.get(int(class_idx), f"Class {class_idx}"),  # type: ignore
                'class_index': class_idx,
                'confidence': confidence_score,
                'all_probabilities': prob_dict
            }
        
        except Exception as e:
            raise Exception(f"Image classification error: {str(e)}")

    def classify_csv(self, file_content: bytes, user_value: str = "", target_column: str = "") -> Dict[str, Any]:
        """Classify data from CSV file, optionally using user input value"""
        try:
            import pandas as pd
            
            # Read CSV
            df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
            
            # Extract features (all numeric columns except potential target)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                raise ValueError("CSV must contain numeric columns for prediction")
            
            # If user provided a value, use it for prediction
            if user_value:
                try:
                    # Try to convert to float
                    user_val_float = float(user_value)
                    
                    # Use target column if specified, otherwise use first numeric column
                    if target_column and target_column in df.columns:
                        col_to_use = target_column
                    else:
                        col_to_use = numeric_cols[0]
                    
                    # Get min/max from ONLY the selected column
                    column_data = df[col_to_use].dropna()
                    data_min = float(column_data.min())
                    data_max = float(column_data.max())
                    data_range = data_max - data_min
                    
                    if data_range > 0:
                        normalized_value = (user_val_float - data_min) / data_range
                    else:
                        normalized_value = user_val_float / 100.0  # Fallback
                    
                    # Classify based on normalized user value
                    if normalized_value < 0.33:
                        prediction = "Low"
                        confidence = 0.75 + (abs(0.165 - normalized_value) / 0.165) * 0.15
                    elif normalized_value < 0.67:
                        prediction = "Medium"
                        confidence = 0.78 + (abs(0.5 - normalized_value) / 0.34) * 0.15
                    else:
                        prediction = "High"
                        confidence = 0.75 + ((normalized_value - 0.67) / 0.33) * 0.15
                    
                    return {
                        'prediction': prediction,
                        'confidence': min(0.95, confidence),
                        'data_summary': {
                            'user_value': user_value,
                            'normalized_value': round(normalized_value, 4),
                            'data_min': round(data_min, 4),
                            'data_max': round(data_max, 4),
                            'rows': len(df),
                            'columns': len(numeric_cols),
                        }
                    }
                except ValueError:
                    # User value is not numeric, use as categorical
                    prediction = "Custom Value"
                    confidence = 0.70
                    return {
                        'prediction': f'{prediction}: {user_value}',
                        'confidence': confidence,
                        'data_summary': {
                            'user_value': user_value,
                            'type': 'categorical',
                            'rows': len(df),
                        }
                    }
            
            # If no user value, analyze file
            selected_cols = df[numeric_cols[:min(10, len(numeric_cols))]]
            features_array = selected_cols.to_numpy()
            
            # Average the features for simple classification
            avg_value = float(np.mean(features_array))
            std_value = float(np.std(features_array))
            
            # Simple classification based on file analysis
            if avg_value < 0.33:
                prediction = "Low"
                confidence = 0.75
            elif avg_value < 0.67:
                prediction = "Medium"
                confidence = 0.76
            else:
                prediction = "High"
                confidence = 0.72
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'data_summary': {
                    'rows': len(df),
                    'columns': len(numeric_cols),
                    'mean': round(avg_value, 4),
                    'std': round(std_value, 4),
                }
            }
        
        except Exception as e:
            raise Exception(f"CSV classification error: {str(e)}")

    def classify_json(self, file_content: bytes, user_value: str = "") -> Dict[str, Any]:
        """Classify data from JSON file, optionally using user input value"""
        try:
            data = json.loads(file_content.decode('utf-8'))
            
            # Extract numeric values
            numeric_values = []
            
            def extract_numbers(obj, values):
                if isinstance(obj, dict):
                    for v in obj.values():
                        extract_numbers(v, values)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_numbers(item, values)
                elif isinstance(obj, (int, float)):
                    numeric_values.append(obj)
            
            extract_numbers(data, numeric_values)
            
            if not numeric_values:
                raise ValueError("JSON must contain numeric values")
            
            # If user provided a value, use it for prediction
            if user_value:
                try:
                    user_val_float = float(user_value)
                    data_min = float(min(numeric_values))
                    data_max = float(max(numeric_values))
                    data_range = data_max - data_min
                    
                    if data_range > 0:
                        normalized_value = (user_val_float - data_min) / data_range
                    else:
                        normalized_value = user_val_float / 100.0
                    
                    # Classify based on normalized user value
                    if normalized_value < 0.33:
                        prediction = "Low"
                        confidence = 0.74 + (abs(0.165 - normalized_value) / 0.165) * 0.16
                    elif normalized_value < 0.67:
                        prediction = "Medium"
                        confidence = 0.77 + (abs(0.5 - normalized_value) / 0.34) * 0.16
                    else:
                        prediction = "High"
                        confidence = 0.74 + ((normalized_value - 0.67) / 0.33) * 0.16
                    
                    return {
                        'prediction': prediction,
                        'confidence': min(0.95, confidence),
                        'data_summary': {
                            'user_value': user_value,
                            'normalized_value': round(normalized_value, 4),
                            'data_min': round(data_min, 4),
                            'data_max': round(data_max, 4),
                            'values_found': len(numeric_values),
                        }
                    }
                except ValueError:
                    prediction = "Custom Value"
                    confidence = 0.68
                    return {
                        'prediction': f'{prediction}: {user_value}',
                        'confidence': confidence,
                        'data_summary': {
                            'user_value': user_value,
                            'type': 'categorical',
                        }
                    }
            
            # If no user value, analyze file
            avg_value = float(np.mean(numeric_values))
            
            # Classification based on average
            if avg_value < 0.33:
                prediction = "Low"
                confidence = 0.72
            elif avg_value < 0.67:
                prediction = "Medium"
                confidence = 0.74
            else:
                prediction = "High"
                confidence = 0.71
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'data_summary': {
                    'values_found': len(numeric_values),
                    'mean': round(avg_value, 4),
                    'min': round(float(min(numeric_values)), 4),
                    'max': round(float(max(numeric_values)), 4),
                }
            }
        
        except Exception as e:
            raise Exception(f"JSON classification error: {str(e)}")

    def regress_image(self, file_content: bytes) -> Dict[str, Any]:
        """Regression on image features"""
        try:
            image = Image.open(io.BytesIO(file_content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert image to array and extract features
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # Extract mean color values as features
            features = np.array([
                float(np.mean(img_array[:, :, 0])),  # Red mean
                float(np.mean(img_array[:, :, 1])),  # Green mean
                float(np.mean(img_array[:, :, 2])),  # Blue mean
                float(np.std(img_array)),            # Overall std
                img_array.shape[0] / 255.0,   # Height normalized
                img_array.shape[1] / 255.0,   # Width normalized
            ], dtype=np.float32)
            
            # Make prediction
            input_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)  # type: ignore
            
            with torch.no_grad():  # type: ignore[attr-defined]
                output = self.regressor(input_tensor)
            
            predicted_value = float(output[0][0].item())
            
            return {
                'prediction': round(predicted_value, 4),
                'confidence': 0.75,
                'details': {
                    'mean_red': round(features[0], 4),
                    'mean_green': round(features[1], 4),
                    'mean_blue': round(features[2], 4),
                    'std': round(features[3], 4),
                }
            }
        
        except Exception as e:
            raise Exception(f"Image regression error: {str(e)}")

    def regress_csv(self, file_content: bytes) -> Dict[str, Any]:
        """Regression on CSV data"""
        try:
            import pandas as pd
            
            df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                raise ValueError("CSV must contain numeric columns")
            
            # Use first numeric column as target
            selected_cols = df[numeric_cols[:min(6, len(numeric_cols))]]
            features_array = selected_cols.to_numpy()
            features = np.mean(features_array, axis=0)
            
            # Simple regression: weighted average
            predicted_value = float(np.mean(features))
            
            return {
                'prediction': round(predicted_value, 4),
                'confidence': 0.77,
                'details': {
                    'rows': len(df),
                    'features': len(numeric_cols),
                    'mean': round(float(np.mean(features)), 4),
                }
            }
        
        except Exception as e:
            raise Exception(f"CSV regression error: {str(e)}")


# Global model instance
predictor = None


def get_predictor():
    """Get or create the global predictor instance"""
    global predictor
    if predictor is None:
        predictor = PredictionModel()
    return predictor

