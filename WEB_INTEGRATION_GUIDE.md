# Integration Guide: Web Learning with Your AI Models

This guide explains how to integrate web scraping and learning capabilities with your existing AI models (UNet, Parking Segmenter, etc.).

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│           Frontend (Next.js)                        │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────▼────────────┬──────────────┐
         │                        │              │
    ┌────▼─────┐            ┌────▼──────┐  ┌──▼──────┐
    │   UNet   │            │ Parking   │  │  Web    │
    │ Detector │            │Segmenter  │  │Learning │
    └──────────┘            └───────────┘  └─────────┘
         │                        │              │
         └────────────┬──────────┬┴──────────────┘
                      │
              ┌───────▼────────┐
              │  MongoDB       │
              │  Database      │
              └────────────────┘
```

## Key Integration Points

### 1. Unified API Server

Your FastAPI server (`server.py`) now includes:
- **Existing Endpoints**: UNet, Parking Segmentation, Training
- **New Web Learning Endpoints**: Scraping, NLP Analysis, Learning Data
- **Shared Infrastructure**: MongoDB, Authentication, CORS

### 2. Data Flow

```
Website → WebScraper → Extracted Content → WebLearner → Learning Data
                            ↓                              ↓
                         MongoDB ←────────────────────────┘
```

### 3. Training Loop Enhancement

You can now use web-scraped data to:
1. Collect training examples from websites
2. Extract and label features using web learning
3. Feed into your existing model training pipeline

## Integration Patterns

### Pattern 1: Scrape and Train

```python
from web_scraper import WebScraper
from web_learner import WebLearner

# Scrape training data from web
scraper = WebScraper()
urls = ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
results = scraper.scrape_multiple_urls(urls)

# Extract features
learner = WebLearner()
for result in results:
    if result['success']:
        learning = learner.extract_learning_data(result['text'])
        
        # Use extracted data for training
        # feeding into your UNet or other models
```

### Pattern 2: Knowledge-Aware Model Training

```python
# Use web learning to enhance model understanding
from web_learner import WebLearner

learner = WebLearner()

# Before training, get context from web
context = learner.extract_learning_data(training_description)

# Apply learned knowledge to model configuration
keywords = context['key_terms']
entities = context['entities']

# Adapt training based on web-learned concepts
print(f"Key concepts: {keywords}")
print(f"Related entities: {entities}")
```

### Pattern 3: Automated Data Annotation

```python
# Use web learning to automatically annotate training data
from web_scraper import WebScraper
from web_learner import WebLearner

def annotate_training_data(image_description_url):
    scraper = WebScraper()
    learner = WebLearner()
    
    # Scrape description
    result = scraper.scrape_url(image_description_url)
    
    if result['success']:
        # Extract learning data for annotation
        learning = learner.extract_learning_data(result['text'])
        
        # Generate annotations
        annotations = {
            'keywords': learning['key_terms'],
            'objects': [e['text'] for e in learning['entities']],
            'summary': learning['facts']
        }
        
        return annotations
```

### Pattern 4: Real-time Model Adaptation

```python
# Learn from web to adapt model behavior
from web_learner import WebLearner
import torch

learner = WebLearner()

def update_model_context(news_url):
    """Update model based on latest web information"""
    
    # Get latest information
    learning = learner.extract_learning_data(get_content_from_url(news_url))
    
    # Extract insights
    keywords = {kw['keyword']: kw['score'] for kw in learning['keywords']}
    
    # Could be used to:
    # - Adjust model weights
    # - Change detection thresholds
    # - Modify post-processing
    
    return keywords
```

## Workflow Examples

### Example 1: Build Training Dataset from Web

```python
from web_scraper import WebScraper
from web_learner import WebLearner
from pymongo import MongoClient
from datetime import datetime

def build_training_dataset(topic_urls):
    """Build a training dataset by scraping and learning from web"""
    
    scraper = WebScraper()
    learner = WebLearner()
    db = MongoClient()['emadaskari_db']
    
    training_data = []
    
    for url in topic_urls:
        # Scrape content
        scrape_result = scraper.scrape_url(url)
        
        if scrape_result['success']:
            # Extract learning data
            learning = learner.extract_learning_data(scrape_result['text'])
            
            # Extract features
            analysis = learner.analyze_content(scrape_result['text'])
            
            # Create training example
            training_example = {
                'source_url': url,
                'raw_text': scrape_result['text'],
                'learning_data': learning,
                'analysis': analysis,
                'timestamp': datetime.now(),
                'status': 'ready_for_review'
            }
            
            training_data.append(training_example)
            
            # Save to database
            db.web_training_data.insert_one(training_example)
            
            print(f"✓ Processed: {url}")
            print(f"  - Entities: {len(learning['entities'])}")
            print(f"  - Key Terms: {len(learning['key_terms'])}")
            print(f"  - Facts: {len(learning['facts'])}")
    
    return training_data
```

### Example 2: Compare Model Predictions with Web Knowledge

```python
def validate_model_with_web_knowledge(model, image, web_knowledge_url):
    """Validate model predictions against web-sourced ground truth"""
    
    from web_learner import WebLearner
    
    # Get model prediction
    model_prediction = model.predict(image)
    
    # Get web-based knowledge
    learner = WebLearner()
    web_data = learner.extract_learning_data(get_content_from_url(web_knowledge_url))
    
    # Extract expected features from web
    expected_features = set(web_data['key_terms'])
    
    # Compare
    validation_result = {
        'model_prediction': model_prediction,
        'web_knowledge': web_data,
        'match_score': calculate_match_score(model_prediction, expected_features)
    }
    
    return validation_result
```

### Example 3: Continuous Learning from Web Updates

```python
from web_scraper import WebScraper
from web_learner import WebLearner
from datetime import datetime, timedelta

def continuous_web_learning(urls, update_interval_hours=24):
    """Continuously learn from web updates"""
    
    scraper = WebScraper()
    learner = WebLearner()
    db = MongoClient()['emadaskari_db']
    
    last_update = datetime.now()
    knowledge_base = {}
    
    while True:
        # Check if update is needed
        if datetime.now() - last_update > timedelta(hours=update_interval_hours):
            
            # Scrape latest content
            for url in urls:
                result = scraper.scrape_url(url)
                
                if result['success']:
                    # Extract learning data
                    learning = learner.extract_learning_data(result['text'])
                    
                    # Update knowledge base
                    key = url.split('/')[-1]
                    knowledge_base[key] = {
                        'learning': learning,
                        'timestamp': datetime.now()
                    }
                    
                    # Save to database
                    db.knowledge_base.update_one(
                        {'key': key},
                        {'$set': knowledge_base[key]},
                        upsert=True
                    )
            
            last_update = datetime.now()
            print(f"✓ Knowledge base updated at {last_update}")
        
        # Could periodically retrain models here
        time.sleep(60)  # Check every minute
```

## Integration with Your Existing Models

### For UNet Model

```python
# Use web learning to understand what UNet should detect
from web_learner import WebLearner

learner = WebLearner()

# Learn what a "haircut line" is
context = learner.extract_learning_data("""
    A haircut line is the boundary between the haircut area and the untouched hair.
    It typically appears as a distinct edge or contour in images.
    The line separates the trimmed portion from the natural hair growth area.
""")

# Extract learned features
detection_features = {
    'keywords': context['key_terms'],
    'definitions': context['definitions'],
    'entities': context['entities']
}

# Could inform model architecture or post-processing
```

### For Parking Segmentation

```python
# Learn parking characteristics from web
from web_scraper import WebScraper
from web_learner import WebLearner

scraper = WebScraper()
learner = WebLearner()

# Scrape parking information
result = scraper.scrape_url("https://example.com/parking-standards")

# Extract parking features
parking_knowledge = learner.extract_learning_data(result['text'])

# Use to improve segmentation
parking_features = parking_knowledge['key_terms']
print(f"Parking features to detect: {parking_features}")
```

## API Workflow

### Client Request Flow

```
1. Frontend requests to analyze content
   POST /web/scrape or /web/analyze
          ↓
2. Backend scrapes website (if scrape endpoint)
          ↓
3. Extract content using BeautifulSoup
          ↓
4. Process with spaCy NLP
          ↓
5. Return structured learning data
          ↓
6. Save to MongoDB for persistence
          ↓
7. Return to Frontend
```

### Example API Usage Flow

```javascript
// Frontend JavaScript Example
async function scrapeAndLearn(url) {
  // Step 1: Scrape the website
  const scrapeResponse = await fetch('http://localhost:8000/web/scrape', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: url })
  });
  
  const scrapeData = await scrapeResponse.json();
  
  if (scrapeData.success) {
    // Step 2: Extract learning data
    const learnResponse = await fetch('http://localhost:8000/web/extract-learning', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: scrapeData.text, url: url })
    });
    
    const learningData = await learnResponse.json();
    return learningData;
  }
}
```

## Performance Considerations

### Scraping Performance
- **Single URL**: 2-5 seconds
- **Batch (10 URLs)**: 20-50 seconds
- **Tip**: Use batch endpoints for multiple URLs

### NLP Processing Performance
- **Small text (< 1KB)**: ~100ms
- **Medium text (10-100KB)**: ~500ms-2s
- **Large text (> 100KB)**: 2-10s
- **Tip**: Process in chunks for very large documents

### Database Performance
- All results automatically saved to MongoDB
- Create indexes on frequently queried fields:

```python
db.web_content.create_index("url")
db.learning_data.create_index("timestamp")
db.content_analysis.create_index("created_at")
```

## Error Handling

### Common Issues and Solutions

1. **Network Errors**
   ```python
   # Automatic retry with backoff
   scraper = WebScraper(max_retries=3, timeout=10)
   ```

2. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Memory Issues**
   - Automatic text limiting in all functions
   - Process large files in chunks

4. **MongoDB Connection**
   - Set MONGO_URI environment variable
   - Check network connectivity

## Deployment Considerations

### Environment Variables

```bash
# MongoDB
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/db

# Server
PORT=8000
ENABLE_PARKING_SEGMENTER=false
```

### Docker Integration

The web learning module is included in the existing Dockerfile:

```dockerfile
# requirements.txt already includes:
beautifulsoup4>=4.11.0
spacy>=3.5.0
requests>=2.28.0
```

### Production Recommendations

1. Use nginx/Gunicorn for production
2. Enable caching for frequently scraped URLs
3. Set up background jobs for batch operations
4. Monitor API response times
5. Implement rate limiting for public APIs

## Next Steps

1. **Test Installation**
   ```bash
   python test_web_learning.py
   ```

2. **Run Examples**
   ```bash
   python web_learning_example.py
   ```

3. **Start Server**
   ```bash
   python -m uvicorn server:app --reload
   ```

4. **Integrate with Your Models**
   - Identify data sources to scrape
   - Design learning pipeline
   - Implement custom workflows

5. **Monitor and Iterate**
   - Check MongoDB for stored learning data
   - Measure model improvement
   - Refine extraction rules

## Support & Resources

- **Documentation**: See `WEB_LEARNING.md`
- **Quick Start**: See `WEB_LEARNING_QUICKSTART.md`
- **Examples**: Run `python web_learning_example.py`
- **Testing**: Run `python test_web_learning.py`

---

Happy integrating! 🚀

