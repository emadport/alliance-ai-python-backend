# Web Learning & Scraping Module

This document describes the web learning functionality that allows the Alliance AI models to scrape websites, extract information, and learn from web content using BeautifulSoup and spaCy.

## Overview

The web learning module provides two main components:

1. **WebScraper** - Extracts content and structured data from websites
2. **WebLearner** - Processes content using NLP to extract insights and learning data

## Installation

### Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

This includes:
- `beautifulsoup4` - HTML parsing and web scraping
- `spacy` - Natural Language Processing
- `requests` - HTTP requests to websites

### SpaCy Model Setup

For full NLP capabilities, download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

For better performance with larger texts:
```bash
python -m spacy download en_core_web_lg
```

## Architecture

### WebScraper (`web_scraper.py`)

The `WebScraper` class handles fetching and extracting information from websites.

#### Key Methods

- **`fetch_url(url: str) -> Optional[str]`**
  - Fetches HTML content from a URL with automatic retry logic
  - Returns raw HTML or None if failed

- **`extract_text(html: str) -> str`**
  - Extracts clean, readable text from HTML
  - Removes scripts, styles, and unnecessary whitespace

- **`extract_structured_data(html: str, url: str) -> Dict`**
  - Extracts structured information:
    - Title and meta description
    - Headings (h1, h2, h3)
    - Paragraphs
    - Links
    - Tables

- **`scrape_url(url: str) -> Dict`**
  - Complete scraping of a single URL
  - Returns both plain text and structured data

- **`scrape_multiple_urls(urls: List[str]) -> List[Dict]`**
  - Batch scraping multiple URLs
  - Processes each URL and returns results

### WebLearner (`web_learner.py`)

The `WebLearner` class processes text using NLP to extract knowledge and insights.

#### Key Methods

- **`extract_entities(text: str) -> List[Dict]`**
  - Identifies named entities (people, places, organizations, etc.)
  - Returns entity text, type, and position in text

- **`extract_noun_phrases(text: str) -> List[str]`**
  - Extracts noun phrases from text
  - Useful for identifying key concepts

- **`extract_keywords(text: str, top_n: int = 20) -> List[Dict]`**
  - Identifies important keywords using frequency analysis
  - Returns keyword with frequency and score

- **`sentiment_analysis(text: str) -> Dict`**
  - Analyzes overall sentiment of content
  - Returns sentiment (positive/neutral/negative) and score

- **`summarize_text(text: str, num_sentences: int = 3) -> List[str]`**
  - Extracts key sentences for summary
  - Uses frequency-based importance scoring

- **`analyze_content(text: str, url: str = "") -> Dict`**
  - Comprehensive analysis combining all methods
  - Returns complete analysis results

- **`compare_texts(text1: str, text2: str) -> Dict`**
  - Compares two texts for similarity
  - Finds common entities and phrases

- **`extract_learning_data(text: str, url: str = "") -> Dict`**
  - Extracts structured learning data:
    - Definitions (sentences with "is")
    - Facts (significant sentences)
    - Questions (sentences starting with question words)
    - Key entities and terms

## API Endpoints

### Web Scraping Endpoints

#### `POST /web/scrape`
Scrape a single URL

**Request:**
```json
{
  "url": "https://example.com/article"
}
```

**Response:**
```json
{
  "success": true,
  "url": "https://example.com/article",
  "text": "extracted text content...",
  "structured": {
    "title": "Page Title",
    "meta_description": "Meta description",
    "headings": [...],
    "paragraphs": [...],
    "links": [...],
    "tables": [...]
  },
  "timestamp": "2024-11-08T12:34:56.789Z"
}
```

#### `POST /web/scrape-batch`
Scrape multiple URLs at once

**Request:**
```json
{
  "urls": [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3"
  ]
}
```

**Response:**
```json
{
  "success": true,
  "count": 3,
  "results": [...]
}
```

### NLP Analysis Endpoints

#### `POST /web/analyze`
Comprehensive NLP analysis of content

**Request:**
```json
{
  "text": "your content here",
  "url": "https://source-url.com"  // optional
}
```

**Response:**
```json
{
  "url": "https://source-url.com",
  "text_length": 5000,
  "word_count": 850,
  "entities": [...],
  "keywords": [...],
  "noun_phrases": [...],
  "sentiment": {
    "sentiment": "positive",
    "score": 0.72
  },
  "summary": [...],
  "success": true,
  "timestamp": "2024-11-08T12:34:56.789Z"
}
```

#### `POST /web/extract-learning`
Extract structured learning data

**Request:**
```json
{
  "text": "your educational content here",
  "url": "https://source.com"  // optional
}
```

**Response:**
```json
{
  "url": "https://source.com",
  "definitions": [
    "Machine learning is a subset of artificial intelligence...",
    ...
  ],
  "facts": [...],
  "questions": [...],
  "entities": [...],
  "key_terms": [...],
  "success": true,
  "timestamp": "2024-11-08T12:34:56.789Z"
}
```

#### `POST /web/entities`
Extract named entities

**Request:**
```json
{
  "text": "Steve Jobs founded Apple in 1976 in California."
}
```

**Response:**
```json
{
  "success": true,
  "entities": [
    {"text": "Steve Jobs", "label": "PERSON", "start": 0, "end": 11},
    {"text": "Apple", "label": "ORG", "start": 22, "end": 27},
    {"text": "1976", "label": "DATE", "start": 31, "end": 35},
    {"text": "California", "label": "GPE", "start": 39, "end": 49}
  ],
  "count": 4
}
```

#### `POST /web/keywords`
Extract keywords from text

**Request:**
```json
{
  "text": "your text content..."
}
```

**Response:**
```json
{
  "success": true,
  "keywords": [
    {"keyword": "machine learning", "frequency": 15, "score": 1.0},
    {"keyword": "artificial intelligence", "frequency": 12, "score": 0.8},
    ...
  ],
  "count": 20
}
```

#### `POST /web/summarize`
Generate summary of text

**Request:**
```json
{
  "text": "long text content..."
}
```

**Response:**
```json
{
  "success": true,
  "summary": [
    "First key sentence...",
    "Second key sentence...",
    "Third key sentence...",
    "Fourth key sentence...",
    "Fifth key sentence..."
  ],
  "sentence_count": 5
}
```

#### `POST /web/sentiment`
Analyze sentiment

**Request:**
```json
{
  "text": "I absolutely love this product! It's amazing."
}
```

**Response:**
```json
{
  "success": true,
  "sentiment": {
    "sentiment": "positive",
    "score": 0.85,
    "token_count": 42
  }
}
```

#### `POST /web/compare`
Compare two texts

**Request:**
```json
{
  "text1": "Machine learning is a subset of AI that enables computers to learn from data.",
  "text2": "Artificial intelligence includes machine learning, which allows systems to improve from experience."
}
```

**Response:**
```json
{
  "similarity_score": 0.82,
  "common_entities": [...],
  "common_phrases": [...],
  "unique_entities_text1": [...],
  "unique_entities_text2": [...],
  "success": true
}
```

### History & Storage

#### `GET /web/history?limit=10`
Get recent scraping history

**Response:**
```json
{
  "success": true,
  "count": 10,
  "history": [
    {
      "url": "https://example.com",
      "timestamp": "2024-11-08T12:34:56.789Z",
      ...
    }
  ]
}
```

#### `GET /web/learning-history?limit=10`
Get recent learning data

**Response:**
```json
{
  "success": true,
  "count": 10,
  "history": [
    {
      "url": "https://example.com",
      "definitions": [...],
      "facts": [...],
      ...
    }
  ]
}
```

## Usage Examples

### Example 1: Scrape and Learn from a Website

```python
from web_scraper import WebScraper
from web_learner import WebLearner

# Initialize
scraper = WebScraper()
learner = WebLearner()

# Scrape a website
scrape_result = scraper.scrape_url("https://en.wikipedia.org/wiki/Machine_learning")

if scrape_result['success']:
    text = scrape_result['text']
    
    # Extract learning data
    learning = learner.extract_learning_data(text)
    
    print("Key Terms:", learning['key_terms'])
    print("Definitions:", learning['definitions'])
    print("Facts:", learning['facts'])
```

### Example 2: Batch Scraping and Analysis

```python
urls = [
    "https://example.com/article1",
    "https://example.com/article2",
    "https://example.com/article3"
]

# Scrape all URLs
results = scraper.scrape_multiple_urls(urls)

# Analyze each one
for result in results:
    if result['success']:
        analysis = learner.analyze_content(result['text'])
        print(f"URL: {result['url']}")
        print(f"Keywords: {[kw['keyword'] for kw in analysis['keywords'][:5]]}")
        print(f"Summary: {analysis['summary']}")
        print("---")
```

### Example 3: Compare Content from Multiple Sources

```python
# Scrape two related articles
result1 = scraper.scrape_url("https://source1.com/article")
result2 = scraper.scrape_url("https://source2.com/article")

if result1['success'] and result2['success']:
    # Compare them
    comparison = learner.compare_texts(result1['text'], result2['text'])
    
    print(f"Similarity: {comparison['similarity_score']}")
    print(f"Common Topics: {comparison['common_phrases']}")
    print(f"Different Aspects: Text1 - {comparison['unique_entities_text1']}")
```

## Best Practices

1. **Rate Limiting**: When scraping multiple sites, add delays to respect server load
2. **User Agent**: The scraper includes a proper User-Agent header
3. **Error Handling**: All functions handle network errors gracefully
4. **Text Size Limits**: NLP processing has built-in text limits to prevent memory issues
5. **MongoDB Storage**: All results are automatically saved to MongoDB for persistence

## Performance Considerations

- **Scraping**: Typical scraping takes 2-5 seconds per URL
- **NLP Processing**: Depends on text length:
  - 1,000 words: ~500ms
  - 10,000 words: ~2-3s
  - 100,000+ words: May need optimization
- **Batch Operations**: Process URLs in parallel for better throughput

## Troubleshooting

### SpaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

### Connection Timeouts
- Increase timeout in WebScraper: `scraper = WebScraper(timeout=15)`
- Check network connectivity to target website

### Memory Issues with Large Texts
- Use text limiting in the learner (already built-in)
- Process large documents in chunks

### MongoDB Connection Issues
- Verify MONGO_URI environment variable
- Check network connectivity to MongoDB

## Future Enhancements

- Sentiment analysis with transformer models (BERT, RoBERTa)
- Multi-language support
- Image extraction and OCR
- PDF scraping capability
- Content classification and categorization
- Question-answering from extracted content
- Knowledge graph construction from multiple sources

