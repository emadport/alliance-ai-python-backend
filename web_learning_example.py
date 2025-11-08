#!/usr/bin/env python3
"""
Web Learning Module - Quick Start Examples
Demonstrates how to use the web scraper and learner
"""

from web_scraper import WebScraper
from web_learner import WebLearner
import json


def example_1_scrape_single_url():
    """Example 1: Scrape a single URL and extract basic information"""
    print("=" * 60)
    print("EXAMPLE 1: Scraping a Single URL")
    print("=" * 60)
    
    scraper = WebScraper()
    
    # Scrape a website
    url = "https://www.python.org"
    print(f"\nScraping: {url}")
    
    result = scraper.scrape_url(url)
    
    if result['success']:
        print(f"\n✓ Successfully scraped!")
        print(f"Title: {result['structured']['title']}")
        print(f"Meta Description: {result['structured']['meta_description']}")
        print(f"\nExtracted Headings:")
        for heading in result['structured']['headings'][:3]:
            print(f"  {heading['level']}: {heading['text']}")
        
        print(f"\nExtracted Links (first 5):")
        for link in result['structured']['links'][:5]:
            print(f"  - {link['text']}: {link['url']}")
        
        print(f"\nText Preview (first 300 chars):")
        print(f"  {result['text'][:300]}...")
    else:
        print(f"✗ Failed to scrape: {result.get('error', 'Unknown error')}")


def example_2_batch_scraping():
    """Example 2: Scrape multiple URLs"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Scraping Multiple URLs")
    print("=" * 60)
    
    scraper = WebScraper()
    
    urls = [
        "https://www.wikipedia.org",
        "https://www.github.com",
    ]
    
    print(f"\nScraping {len(urls)} URLs...")
    results = scraper.scrape_multiple_urls(urls)
    
    for result in results:
        if result['success']:
            print(f"\n✓ {result['url']}")
            print(f"  Text length: {len(result['text'])} characters")
            print(f"  Title: {result['structured']['title']}")
        else:
            print(f"\n✗ {result['url']}: {result.get('error', 'Unknown error')}")


def example_3_nlp_analysis():
    """Example 3: NLP analysis of content"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: NLP Analysis of Content")
    print("=" * 60)
    
    learner = WebLearner()
    
    if not learner.nlp:
        print("\n✗ spaCy model not loaded. Install with:")
        print("  python -m spacy download en_core_web_sm")
        return
    
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines. 
    AI has applications in multiple fields including healthcare, finance, and transportation. 
    Machine learning is a subset of artificial intelligence that enables 
    systems to learn and improve from experience without being explicitly programmed. 
    Deep learning is a branch of machine learning using multi-layer neural networks.
    """
    
    print("\nAnalyzing text:")
    print(f"'{text[:100]}...'")
    
    analysis = learner.analyze_content(text)
    
    print(f"\n📊 Analysis Results:")
    print(f"  Text Length: {analysis['text_length']} characters")
    print(f"  Word Count: {analysis['word_count']} words")
    
    print(f"\n🏷️ Entities Found:")
    for entity in analysis['entities'][:5]:
        print(f"  - {entity['text']} ({entity['label']})")
    
    print(f"\n🔑 Keywords:")
    for keyword in analysis['keywords'][:5]:
        print(f"  - {keyword['keyword']} (score: {keyword['score']:.2f})")
    
    print(f"\n📝 Summary:")
    for i, sentence in enumerate(analysis['summary'], 1):
        print(f"  {i}. {sentence}")
    
    print(f"\n😊 Sentiment: {analysis['sentiment']['sentiment']} (score: {analysis['sentiment']['score']:.2f})")


def example_4_extract_learning_data():
    """Example 4: Extract structured learning data"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Extract Structured Learning Data")
    print("=" * 60)
    
    learner = WebLearner()
    
    if not learner.nlp:
        print("\n✗ spaCy model not loaded.")
        return
    
    text = """
    Python is a high-level programming language. Python was created by Guido van Rossum 
    and first released in 1991. What is Python used for? Python is used for web development, 
    data science, artificial intelligence, and many other applications. Machine learning is 
    a field within artificial intelligence. Why use Python? Python has a large standard library 
    and many third-party packages available through PyPI.
    """
    
    print("\nExtracting learning data from text...")
    learning = learner.extract_learning_data(text)
    
    print(f"\n📚 Definitions Found:")
    for definition in learning['definitions'][:3]:
        print(f"  - {definition}")
    
    print(f"\n📌 Key Terms:")
    for term in learning['key_terms'][:8]:
        print(f"  - {term}")
    
    print(f"\n❓ Questions Found:")
    for question in learning['questions'][:3]:
        print(f"  - {question}")


def example_5_compare_texts():
    """Example 5: Compare two texts"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Compare Two Texts")
    print("=" * 60)
    
    learner = WebLearner()
    
    if not learner.nlp:
        print("\n✗ spaCy model not loaded.")
        return
    
    text1 = "Apple is a technology company founded by Steve Jobs. Apple makes iPhones, iPads, and Macs."
    text2 = "Microsoft is a software company. Bill Gates founded Microsoft. Microsoft creates Windows, Office, and Azure."
    
    print(f"\nText 1: {text1[:60]}...")
    print(f"Text 2: {text2[:60]}...")
    
    comparison = learner.compare_texts(text1, text2)
    
    print(f"\n🔗 Similarity Score: {comparison['similarity_score']:.2f}")
    print(f"\n🤝 Common Entities:")
    if comparison['common_entities']:
        for entity in comparison['common_entities']:
            print(f"  - {entity}")
    else:
        print("  (none)")
    
    print(f"\n🏢 Unique Entities in Text 1:")
    for entity in comparison['unique_entities_text1'][:3]:
        print(f"  - {entity}")
    
    print(f"\n🏢 Unique Entities in Text 2:")
    for entity in comparison['unique_entities_text2'][:3]:
        print(f"  - {entity}")


def example_6_keyword_extraction():
    """Example 6: Detailed keyword extraction"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Keyword Extraction")
    print("=" * 60)
    
    learner = WebLearner()
    
    if not learner.nlp:
        print("\n✗ spaCy model not loaded.")
        return
    
    text = """
    Machine learning is transforming industries. In healthcare, machine learning helps with 
    diagnosis. In finance, machine learning detects fraud. Natural language processing is used 
    for chatbots. Computer vision helps with image recognition. Deep learning uses neural networks. 
    These AI technologies are growing rapidly.
    """
    
    print("\nExtracting keywords from technical text...")
    keywords = learner.extract_keywords(text, top_n=10)
    
    print(f"\n🔑 Top 10 Keywords:")
    for i, kw in enumerate(keywords, 1):
        print(f"  {i:2d}. {kw['keyword']:20s} | Freq: {kw['frequency']:3d} | Score: {kw['score']:.2f}")


def example_7_sentiment_analysis():
    """Example 7: Sentiment analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Sentiment Analysis")
    print("=" * 60)
    
    learner = WebLearner()
    
    texts = [
        "This product is absolutely amazing! I love it!",
        "This is terrible and I hate it. Worst experience ever.",
        "This is okay. Neither good nor bad.",
    ]
    
    print("\nAnalyzing sentiment of different texts:\n")
    for text in texts:
        sentiment = learner.sentiment_analysis(text)
        print(f"Text: \"{text}\"")
        print(f"  Sentiment: {sentiment['sentiment']}")
        print(f"  Score: {sentiment['score']:.2f}\n")


def example_8_text_summarization():
    """Example 8: Text summarization"""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Text Summarization")
    print("=" * 60)
    
    learner = WebLearner()
    
    if not learner.nlp:
        print("\n✗ spaCy model not loaded.")
        return
    
    text = """
    Climate change is one of the greatest challenges facing humanity today.
    Rising global temperatures are causing sea levels to rise and extreme weather events to become more frequent.
    Scientists have shown that human activities, particularly the burning of fossil fuels, 
    are the primary cause of climate change.
    To address this crisis, we need to transition to renewable energy sources like solar and wind power.
    Governments, businesses, and individuals all have a role to play in reducing carbon emissions.
    The transition to clean energy is not only necessary for environmental protection but also creates new economic opportunities.
    Investing in renewable energy and sustainable practices can help mitigate the worst effects of climate change.
    """
    
    print("\nText to summarize:")
    print(f"  {text[:100]}...\n")
    
    summary = learner.summarize_text(text, num_sentences=3)
    
    print("📋 Summary (3 key sentences):")
    for i, sentence in enumerate(summary, 1):
        print(f"  {i}. {sentence}")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Web Learning Module - Quick Start Examples".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        # Run examples
        example_1_scrape_single_url()
        # example_2_batch_scraping()  # Uncomment to run
        example_3_nlp_analysis()
        example_4_extract_learning_data()
        example_5_compare_texts()
        example_6_keyword_extraction()
        example_7_sentiment_analysis()
        example_8_text_summarization()
        
        print("\n" + "=" * 60)
        print("✓ All examples completed!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

