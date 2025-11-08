# Web Learning Module - Quick Start

Get started with web scraping and NLP-based learning in 5 minutes!

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `beautifulsoup4` - Web scraping
- `spacy` - NLP processing  
- `requests` - HTTP requests

### 2. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

## 5-Minute Quick Start

### Test 1: Scrape a Website

```python
from web_scraper import WebScraper

scraper = WebScraper()
result = scraper.scrape_url("https://www.wikipedia.org")

print("Title:", result['structured']['title'])
print("Text preview:", result['text'][:200])
```

### Test 2: Extract Entities

```python
from web_learner import WebLearner

learner = WebLearner()
text = "Steve Jobs founded Apple in California in 1976."

entities = learner.extract_entities(text)
for entity in entities:
    print(f"{entity['text']} - {entity['label']}")
```

### Test 3: Extract Keywords

```python
from web_learner import WebLearner

learner = WebLearner()
text = "Machine learning is a subset of AI. AI is transforming industries. Machine learning enables predictions."

keywords = learner.extract_keywords(text, top_n=5)
for kw in keywords:
    print(f"{kw['keyword']}: {kw['frequency']} occurrences")
```

### Test 4: Analyze Content

```python
from web_learner import WebLearner

learner = WebLearner()
text = "Your content here..."

analysis = learner.analyze_content(text)
print("Sentiment:", analysis['sentiment'])
print("Summary:", analysis['summary'])
print("Keywords:", [k['keyword'] for k in analysis['keywords'][:5]])
```

## Running the Example Script

```bash
python web_learning_example.py
```

This runs 8 complete examples demonstrating all features.

## Using the API

### Start the Server

```bash
python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Example: Scrape via API

```bash
curl -X POST "http://localhost:8000/web/scrape" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.example.com"}'
```

### Example: Extract Keywords via API

```bash
curl -X POST "http://localhost:8000/web/keywords" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning enables predictions. AI transforms industries.",
    "url": "https://example.com"
  }'
```

## Available API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/web/scrape` | POST | Scrape a single URL |
| `/web/scrape-batch` | POST | Scrape multiple URLs |
| `/web/analyze` | POST | Comprehensive NLP analysis |
| `/web/extract-learning` | POST | Extract learning data (definitions, facts, questions) |
| `/web/entities` | POST | Extract named entities |
| `/web/keywords` | POST | Extract keywords |
| `/web/summarize` | POST | Generate summary |
| `/web/sentiment` | POST | Analyze sentiment |
| `/web/compare` | POST | Compare two texts |
| `/web/history` | GET | Get scraping history |
| `/web/learning-history` | GET | Get learning data history |

## Complete Workflow Example

```python
from web_scraper import WebScraper
from web_learner import WebLearner

# 1. Scrape a website
scraper = WebScraper()
scrape_result = scraper.scrape_url("https://en.wikipedia.org/wiki/Artificial_intelligence")

if scrape_result['success']:
    text = scrape_result['text']
    
    # 2. Extract learning data
    learner = WebLearner()
    learning = learner.extract_learning_data(text)
    
    # 3. Print results
    print("=== Learning Data ===")
    print("\nKey Terms:")
    for term in learning['key_terms'][:10]:
        print(f"  - {term}")
    
    print("\nDefinitions Found:")
    for definition in learning['definitions'][:3]:
        print(f"  - {definition[:100]}...")
    
    print("\nKey Entities:")
    for entity in learning['entities'][:5]:
        print(f"  - {entity['text']} ({entity['label']})")
```

## Common Tasks

### Task 1: Learn from Multiple Sources

```python
urls = [
    "https://source1.com/article",
    "https://source2.com/article",
    "https://source3.com/article"
]

scraper = WebScraper()
results = scraper.scrape_multiple_urls(urls)

learner = WebLearner()
all_learnings = []

for result in results:
    if result['success']:
        learning = learner.extract_learning_data(result['text'])
        all_learnings.append(learning)
```

### Task 2: Identify Similar Articles

```python
text1 = scraper.scrape_url(url1)['text']
text2 = scraper.scrape_url(url2)['text']

comparison = learner.compare_texts(text1, text2)
print(f"Similarity: {comparison['similarity_score']}")
print(f"Common topics: {comparison['common_phrases']}")
```

### Task 3: Build a Knowledge Base

```python
from pymongo import MongoClient

# Store all learning data in MongoDB
for url in urls:
    result = scraper.scrape_url(url)
    if result['success']:
        learning = learner.extract_learning_data(result['text'])
        
        # Save to database
        db.learning_data.insert_one({
            'url': url,
            'learning': learning,
            'timestamp': datetime.now()
        })
```

## Troubleshooting

### spaCy Model Not Found

Error:
```
OSError: [E050] Can't find model 'en_core_web_sm'
```

Solution:
```bash
python -m spacy download en_core_web_sm
```

### Connection Timeout

Increase timeout:
```python
scraper = WebScraper(timeout=15)  # 15 seconds instead of 10
```

### Memory Issues with Large Texts

The modules automatically handle large texts:
- Text extraction limits: 1,000,000 characters
- Sentiment analysis limits: 100,000 characters
- Summarization limits: 200,000 characters

### MongoDB Connection Issues

Check environment variable:
```bash
echo $MONGO_URI
```

## Next Steps

1. ✅ Read `WEB_LEARNING.md` for detailed API documentation
2. ✅ Run `python web_learning_example.py` to see all features
3. ✅ Integrate with your AI models
4. ✅ Build a knowledge base from web sources
5. ✅ Create custom learning pipelines

## Performance Tips

- **Batch Operations**: Process multiple URLs together for efficiency
- **Caching**: Cache scraping results to avoid re-fetching
- **Text Limits**: The system automatically handles large texts efficiently
- **Async Processing**: Use async endpoints for better performance

## API Response Format

All responses include a `success` field:

**Success Response:**
```json
{
  "success": true,
  "data": {...}
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Error description"
}
```

## Support

For more information:
- See `WEB_LEARNING.md` for complete documentation
- See `web_learning_example.py` for code examples
- Check the source files: `web_scraper.py` and `web_learner.py`

---

Happy learning! 🚀

