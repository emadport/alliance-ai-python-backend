"""
Model Persistence Module
Saves and loads trained models for reuse
"""

import os
import json
import pickle
from datetime import datetime
from typing import Any, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPersistence:
    """Save and load models and learned data"""
    
    def __init__(self, model_dir: str = "saved_models"):
        """
        Initialize model persistence
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Model persistence initialized: {model_dir}")
    
    def save_model(self, model: Any, model_name: str, metadata: Optional[Dict] = None) -> str:
        """
        Save a trained model to disk
        
        Args:
            model: Model object to save
            model_name: Name for the model
            metadata: Optional metadata about the model
            
        Returns:
            Path to saved model
        """
        try:
            filepath = os.path.join(self.model_dir, f"{model_name}.pkl")
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            meta = {
                'name': model_name,
                'saved_at': datetime.now().isoformat(),
                'file_path': filepath,
                **(metadata or {})
            }
            
            meta_filepath = os.path.join(self.model_dir, f"{model_name}_meta.json")
            with open(meta_filepath, 'w') as f:
                json.dump(meta, f, indent=2)
            
            logger.info(f"Model saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load a saved model from disk
        
        Args:
            model_name: Name of model to load
            
        Returns:
            Loaded model or None if not found
        """
        try:
            filepath = os.path.join(self.model_dir, f"{model_name}.pkl")
            
            if not os.path.exists(filepath):
                logger.warning(f"Model not found: {filepath}")
                return None
            
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Model loaded: {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def save_learning_data(self, data: Dict, data_name: str) -> str:
        """
        Save extracted learning data
        
        Args:
            data: Learning data dictionary
            data_name: Name for the data
            
        Returns:
            Path to saved data
        """
        try:
            filepath = os.path.join(self.model_dir, f"{data_name}.json")
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Learning data saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving learning data: {str(e)}")
            raise
    
    def load_learning_data(self, data_name: str) -> Optional[Dict]:
        """
        Load saved learning data
        
        Args:
            data_name: Name of data to load
            
        Returns:
            Loaded data or None if not found
        """
        try:
            filepath = os.path.join(self.model_dir, f"{data_name}.json")
            
            if not os.path.exists(filepath):
                logger.warning(f"Learning data not found: {filepath}")
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Learning data loaded: {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading learning data: {str(e)}")
            return None
    
    def list_models(self) -> list:
        """Get list of all saved models"""
        try:
            models = []
            for file in os.listdir(self.model_dir):
                if file.endswith('.pkl'):
                    model_name = file.replace('.pkl', '')
                    meta_file = os.path.join(self.model_dir, f"{model_name}_meta.json")
                    
                    meta = None
                    if os.path.exists(meta_file):
                        with open(meta_file, 'r') as f:
                            meta = json.load(f)
                    
                    models.append({
                        'name': model_name,
                        'file': file,
                        'metadata': meta
                    })
            
            return models
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def list_learning_data(self) -> list:
        """Get list of all saved learning data"""
        try:
            data_files = []
            for file in os.listdir(self.model_dir):
                if file.endswith('.json') and not file.endswith('_meta.json'):
                    data_name = file.replace('.json', '')
                    filepath = os.path.join(self.model_dir, file)
                    file_size = os.path.getsize(filepath)
                    
                    data_files.append({
                        'name': data_name,
                        'file': file,
                        'size_kb': file_size / 1024
                    })
            
            return data_files
        except Exception as e:
            logger.error(f"Error listing learning data: {str(e)}")
            return []
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a saved model"""
        try:
            filepath = os.path.join(self.model_dir, f"{model_name}.pkl")
            meta_filepath = os.path.join(self.model_dir, f"{model_name}_meta.json")
            
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Model deleted: {filepath}")
            
            if os.path.exists(meta_filepath):
                os.remove(meta_filepath)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting model: {str(e)}")
            return False
    
    def delete_learning_data(self, data_name: str) -> bool:
        """Delete saved learning data"""
        try:
            filepath = os.path.join(self.model_dir, f"{data_name}.json")
            
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Learning data deleted: {filepath}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error deleting learning data: {str(e)}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get metadata about a saved model"""
        try:
            meta_filepath = os.path.join(self.model_dir, f"{model_name}_meta.json")
            
            if not os.path.exists(meta_filepath):
                return None
            
            with open(meta_filepath, 'r') as f:
                meta = json.load(f)
            
            # Add file size
            filepath = os.path.join(self.model_dir, f"{model_name}.pkl")
            if os.path.exists(filepath):
                meta['size_mb'] = os.path.getsize(filepath) / (1024 * 1024)
            
            return meta
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return None


class KnowledgeBase:
    """Persistent knowledge base from web learning"""
    
    def __init__(self, db_path: str = "knowledge_base"):
        """Initialize knowledge base"""
        self.db_path = db_path
        self.persistence = ModelPersistence(db_path)
        self.knowledge = {}
    
    def add_knowledge(self, topic: str, learning_data: Dict) -> bool:
        """
        Add learning data to knowledge base
        
        Args:
            topic: Topic/subject name
            learning_data: Extracted learning data
            
        Returns:
            Success status
        """
        try:
            self.knowledge[topic] = {
                **learning_data,
                'added_at': datetime.now().isoformat()
            }
            
            # Save to disk
            self.persistence.save_learning_data(self.knowledge[topic], topic)
            logger.info(f"Knowledge added: {topic}")
            return True
        except Exception as e:
            logger.error(f"Error adding knowledge: {str(e)}")
            return False
    
    def get_knowledge(self, topic: str) -> Optional[Dict]:
        """Get knowledge about a topic"""
        if topic in self.knowledge:
            return self.knowledge[topic]
        
        # Try to load from disk
        data = self.persistence.load_learning_data(topic)
        if data:
            self.knowledge[topic] = data
        
        return data
    
    def search_knowledge(self, keyword: str) -> list:
        """Search knowledge base for keyword"""
        results = []
        
        for topic, data in self.knowledge.items():
            # Search in key terms
            key_terms = data.get('key_terms', [])
            if any(keyword.lower() in term.lower() for term in key_terms):
                results.append({
                    'topic': topic,
                    'matching_terms': [t for t in key_terms if keyword.lower() in t.lower()]
                })
            
            # Search in definitions
            definitions = data.get('definitions', [])
            if any(keyword.lower() in d.lower() for d in definitions):
                results.append({
                    'topic': topic,
                    'matching_definitions': [d for d in definitions if keyword.lower() in d.lower()]
                })
        
        return results
    
    def list_topics(self) -> list:
        """List all topics in knowledge base"""
        data_files = self.persistence.list_learning_data()
        return [d['name'] for d in data_files]


# Example usage
if __name__ == "__main__":
    # Initialize persistence
    persistence = ModelPersistence()
    
    # Save learning data
    learning = {
        'url': 'https://example.com',
        'key_terms': ['machine learning', 'AI', 'deep learning'],
        'definitions': ['ML is a type of AI'],
        'facts': ['ML algorithms learn from data']
    }
    
    persistence.save_learning_data(learning, 'machine_learning')
    
    # Load it back
    loaded = persistence.load_learning_data('machine_learning')
    print("Loaded data:", loaded)
    
    # List all saved data
    print("Saved learning data:", persistence.list_learning_data())
    
    # Knowledge base example
    kb = KnowledgeBase()
    kb.add_knowledge('AI', learning)
    
    print("Topics:", kb.list_topics())
    print("AI Knowledge:", kb.get_knowledge('AI'))

