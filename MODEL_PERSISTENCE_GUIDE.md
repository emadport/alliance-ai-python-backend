# Model Persistence & Knowledge Base Guide

## Overview

Your AI models now support **saving, loading, and reusing** learned data through:

1. **Model Persistence** - Save/load trained models to disk
2. **Knowledge Base** - Persistent storage of extracted learning data
3. **MongoDB** - Long-term storage of all scraped and analyzed content

---

## What Gets Saved?

### Automatically Saved (MongoDB):

- ✅ Scraped website content (`web_content` collection)
- ✅ NLP analysis results (`content_analysis` collection)
- ✅ Extracted learning data (`learning_data` collection)

### File-Based (Knowledge Base):

- ✅ Topic-specific learning data (JSON files)
- ✅ Model metadata
- ✅ Training artifacts

---

## Saving Learning Data

### Method 1: Save to Knowledge Base

```python
from model_persistence import KnowledgeBase

kb = KnowledgeBase()

# Add learning data about a topic
kb.add_knowledge('machine_learning', {
    'key_terms': ['ML', 'training', 'model'],
    'definitions': ['Machine learning is...'],
    'facts': ['ML requires data...']
})
```

### Method 2: API Endpoint

```bash
# Save learning data to knowledge base
POST /knowledge/machine_learning
{
  "text": "Machine learning is a type of AI that...",
  "url": "https://source.com"
}
```

---

## Loading & Reusing Data

### Method 1: Load from Knowledge Base

```python
kb = KnowledgeBase()

# Get knowledge about a topic
knowledge = kb.get_knowledge('machine_learning')

# Get list of all topics
topics = kb.list_topics()

# Search for keyword
results = kb.search_knowledge('learning')
```

### Method 2: API Endpoints

```bash
# Get all topics
GET /knowledge/topics

# Get knowledge about specific topic
GET /knowledge/machine_learning

# Search for keyword
GET /knowledge/search/learning
```

---

## Retrieving Historical Data

### From MongoDB (Web Content)

```bash
# Get recent scraped content
GET /web/history?limit=10

# Get recent learning data
GET /web/learning-history?limit=10
```

### From Knowledge Base

```bash
# List all saved knowledge topics
GET /knowledge/topics

# Get specific topic knowledge
GET /knowledge/python_programming
```

---

## Complete Workflow Example

### Step 1: Learn from Multiple Sources

```python
from web_scraper import WebScraper
from web_learner import WebLearner
from model_persistence import KnowledgeBase

scraper = WebScraper()
learner = WebLearner()
kb = KnowledgeBase()

# Scrape and learn from multiple sources
urls = [
    "https://source1.com/article",
    "https://source2.com/article"
]

for url in urls:
    result = scraper.scrape_url(url)
    if result['success']:
        # Extract learning data
        learning = learner.extract_learning_data(result['text'])

        # Save to knowledge base
        kb.add_knowledge('my_topic', learning)
```

### Step 2: Retrieve and Reuse Later

```python
# Later, load the saved knowledge
knowledge = kb.get_knowledge('my_topic')

# Use the learned terms
key_terms = knowledge['key_terms']
definitions = knowledge['definitions']

# Reuse for training or analysis
print(f"Key concepts: {key_terms}")
```

### Step 3: Search and Integrate

```python
# Search knowledge base
results = kb.search_knowledge('specific_keyword')

# Use results in your models
for result in results:
    topic = result['topic']
    matching_terms = result['matching_terms']
    # ... use in training, prediction, etc.
```

---

## New API Endpoints

### Knowledge Base Endpoints

| Endpoint                      | Method | Purpose                     |
| ----------------------------- | ------ | --------------------------- |
| `/knowledge/topics`           | GET    | List all topics             |
| `/knowledge/{topic}`          | GET    | Get knowledge about a topic |
| `/knowledge/{topic}`          | POST   | Save learning data to topic |
| `/knowledge/search/{keyword}` | GET    | Search for keyword          |

### Model Management Endpoints

| Endpoint               | Method | Purpose               |
| ---------------------- | ------ | --------------------- |
| `/models/list`         | GET    | List all saved models |
| `/models/{model_name}` | GET    | Get model information |
| `/models/{model_name}` | DELETE | Delete a saved model  |

---

## File Structure

```
allianceai-backend/
├── knowledge_base/          # Persistent knowledge storage
│   ├── machine_learning.json
│   ├── python_programming.json
│   └── ...
├── saved_models/            # Saved trained models
│   ├── model1.pkl
│   ├── model1_meta.json
│   └── ...
├── model_persistence.py     # Persistence module
└── server.py               # API with new endpoints
```

---

## Usage Examples

### Example 1: Save and Reuse Learning Data

```python
# Save learning about Python
POST /knowledge/python
Content-Type: application/json
{
  "text": "Python is a high-level programming language used for web development, data science, and AI.",
  "url": "https://python.org"
}

# Later, retrieve it
GET /knowledge/python

# Get all topics you've learned about
GET /knowledge/topics
```

### Example 2: Search Knowledge Base

```python
# Save knowledge about multiple AI topics
POST /knowledge/machine_learning {...}
POST /knowledge/deep_learning {...}
POST /knowledge/natural_language_processing {...}

# Search for all knowledge about "learning"
GET /knowledge/search/learning

# Returns: All topics with "learning" in key terms
```

### Example 3: Integrate with UNet Training

```python
from model_persistence import KnowledgeBase

kb = KnowledgeBase()

# Get knowledge about haircut detection
knowledge = kb.get_knowledge('haircut_detection')

# Extract key concepts to guide training
concepts = knowledge['key_terms']
facts = knowledge['facts']

# Use in training pipeline
for concept in concepts:
    print(f"Train model to detect: {concept}")
```

---

## Data Persistence Strategy

### MongoDB (Cloud, Synchronized)

- **Best for**: Long-term storage, multi-server access
- **Saved**: Scraped content, analysis results
- **Retention**: Permanent (as long as MongoDB subscription active)

### Knowledge Base (Local Disk)

- **Best for**: Fast access, offline availability
- **Saved**: Extracted learning data, topics
- **Retention**: Persistent until deleted
- **Location**: `knowledge_base/` directory

### Spacy Model (Pre-trained)

- **Best for**: NLP processing
- **Saved**: Language model (downloaded once)
- **Retention**: Until manually deleted
- **Size**: ~40 MB for `en_core_web_sm`

---

## Best Practices

1. **Organize by Topic**

   - Name topics clearly: `machine_learning`, `cv_concepts`, etc.
   - One topic per domain of knowledge

2. **Regular Backups**

   - Backup `knowledge_base/` directory
   - Backup MongoDB collections

3. **Search Before Saving**

   - Check if knowledge already exists
   - Avoid duplicate learning data

4. **Clean Up Old Models**

   - Delete unused models to save space
   - Use `DELETE /models/{model_name}`

5. **Version Control**
   - Track changes with MongoDB timestamps
   - Keep metadata with saved models

---

## Performance Notes

- **Save**: ~10-50ms for JSON save
- **Load**: ~5-20ms for knowledge retrieval
- **Search**: O(n) where n = number of topics
- **Database**: MongoDB handles 1000s of documents efficiently

---

## Troubleshooting

### Knowledge Not Being Saved

```bash
# Check if knowledge base is initialized
GET /knowledge/topics

# If error, restart server
# Make sure knowledge_base/ directory exists
```

### Memory Issues with Large Models

- Split large models into parts
- Delete unused models: `DELETE /models/old_model`
- Use MongoDB for large-scale storage instead

### Can't Find Saved Knowledge

```bash
# Search for it
GET /knowledge/search/keyword_from_topic

# List all topics
GET /knowledge/topics
```

---

## Integration with Your Models

### For UNet

```python
# Load knowledge about what to detect
kb.get_knowledge('line_detection')
# Use key_terms and facts to guide training
```

### For Parking Segmenter

```python
# Load parking lot characteristics
kb.get_knowledge('parking_segmentation')
# Use facts to improve detection
```

### For Training Pipeline

```python
# Get learning data before training
knowledge = kb.get_knowledge('training_domain')
# Extract definitions and facts
# Use as training guidance
```

---

## Next Steps

1. ✅ Start the backend server
2. ✅ Use `/knowledge/` endpoints to save learning data
3. ✅ Retrieve with `GET /knowledge/{topic}`
4. ✅ Integrate with your AI models
5. ✅ Monitor knowledge base growth: `GET /knowledge/topics`

---

Happy learning and model reuse! 🚀
