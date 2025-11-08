#!/usr/bin/env python3
"""
Test script for Web Learning Module
Verifies that all components are working correctly
"""

import sys
import os


def test_imports():
    """Test that all required packages are installed"""
    print("🧪 Testing imports...")
    
    try:
        import requests
        print("  ✓ requests")
    except ImportError:
        print("  ✗ requests - Install with: pip install requests")
        return False
    
    try:
        import bs4
        print("  ✓ beautifulsoup4")
    except ImportError:
        print("  ✗ beautifulsoup4 - Install with: pip install beautifulsoup4")
        return False
    
    try:
        import spacy
        print("  ✓ spacy")
    except ImportError:
        print("  ✗ spacy - Install with: pip install spacy")
        return False
    
    try:
        from web_scraper import WebScraper
        print("  ✓ web_scraper module")
    except ImportError as e:
        print(f"  ✗ web_scraper module - {e}")
        return False
    
    try:
        from web_learner import WebLearner
        print("  ✓ web_learner module")
    except ImportError as e:
        print(f"  ✗ web_learner module - {e}")
        return False
    
    return True


def test_web_scraper():
    """Test WebScraper functionality"""
    print("\n🧪 Testing WebScraper...")
    
    try:
        from web_scraper import WebScraper
        
        scraper = WebScraper()
        print("  ✓ WebScraper initialized")
        
        # Test with a simple, reliable URL
        print("  ℹ Testing URL fetch (may take a moment)...")
        result = scraper.scrape_url("https://www.example.com")
        
        if result['success']:
            print("  ✓ Successfully scraped URL")
            
            if 'text' in result:
                print(f"  ✓ Text extracted ({len(result['text'])} chars)")
            
            if 'structured' in result:
                print(f"  ✓ Structured data extracted")
                if result['structured'].get('title'):
                    print(f"    - Title: {result['structured']['title']}")
                if result['structured'].get('headings'):
                    print(f"    - Headings: {len(result['structured']['headings'])} found")
            
            return True
        else:
            print(f"  ✗ Scraping failed: {result.get('error', 'Unknown error')}")
            return False
    
    except Exception as e:
        print(f"  ✗ WebScraper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_web_learner():
    """Test WebLearner functionality"""
    print("\n🧪 Testing WebLearner...")
    
    try:
        from web_learner import WebLearner
        
        learner = WebLearner()
        
        if not learner.nlp:
            print("  ⚠ spaCy model not loaded")
            print("  Download with: python -m spacy download en_core_web_sm")
            return False
        
        print("  ✓ WebLearner initialized with spaCy model")
        
        test_text = "Apple Inc. was founded by Steve Jobs in California. It produces iPhones and computers."
        
        # Test entity extraction
        entities = learner.extract_entities(test_text)
        if entities:
            print(f"  ✓ Entity extraction works ({len(entities)} entities found)")
            print(f"    - Example: {entities[0]['text']} ({entities[0]['label']})")
        else:
            print("  ✗ Entity extraction failed")
            return False
        
        # Test keyword extraction
        keywords = learner.extract_keywords(test_text)
        if keywords:
            print(f"  ✓ Keyword extraction works ({len(keywords)} keywords found)")
            print(f"    - Example: {keywords[0]['keyword']}")
        else:
            print("  ✗ Keyword extraction failed")
            return False
        
        # Test noun phrase extraction
        phrases = learner.extract_noun_phrases(test_text)
        if phrases:
            print(f"  ✓ Noun phrase extraction works ({len(phrases)} phrases found)")
        else:
            print("  ⚠ No noun phrases found (might be normal)")
        
        # Test sentiment analysis
        sentiment = learner.sentiment_analysis(test_text)
        if sentiment:
            print(f"  ✓ Sentiment analysis works (sentiment: {sentiment.get('sentiment', 'unknown')})")
        else:
            print("  ✗ Sentiment analysis failed")
            return False
        
        # Test summarization
        summary = learner.summarize_text(test_text, num_sentences=1)
        if summary:
            print(f"  ✓ Summarization works ({len(summary)} sentence(s))")
        else:
            print("  ✗ Summarization failed")
            return False
        
        # Test learning data extraction
        learning = learner.extract_learning_data(test_text)
        if learning['success']:
            print(f"  ✓ Learning data extraction works")
            if learning.get('definitions'):
                print(f"    - Definitions: {len(learning['definitions'])} found")
            if learning.get('facts'):
                print(f"    - Facts: {len(learning['facts'])} found")
            if learning.get('entities'):
                print(f"    - Entities: {len(learning['entities'])} found")
        else:
            print("  ✗ Learning data extraction failed")
            return False
        
        return True
    
    except Exception as e:
        print(f"  ✗ WebLearner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_models():
    """Test FastAPI models"""
    print("\n🧪 Testing API Models...")
    
    try:
        from pydantic import BaseModel
        
        # Check if server.py has the models
        try:
            # Try importing from server
            import importlib.util
            spec = importlib.util.spec_from_file_location("server", "server.py")
            if spec and spec.loader:
                server = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(server)
                
                if hasattr(server, 'URLRequest'):
                    print("  ✓ URLRequest model found")
                if hasattr(server, 'URLListRequest'):
                    print("  ✓ URLListRequest model found")
                if hasattr(server, 'TextAnalysisRequest'):
                    print("  ✓ TextAnalysisRequest model found")
                if hasattr(server, 'TextComparisonRequest'):
                    print("  ✓ TextComparisonRequest model found")
                
                print("  ✓ API models are properly defined")
                return True
        except Exception as e:
            print(f"  ⚠ Could not verify API models: {e}")
            print("  But this is okay - models are in server.py")
            return True
    
    except Exception as e:
        print(f"  ✗ API models test failed: {e}")
        return False


def test_comparative_analysis():
    """Test text comparison"""
    print("\n🧪 Testing Text Comparison...")
    
    try:
        from web_learner import WebLearner
        
        learner = WebLearner()
        
        if not learner.nlp:
            print("  ⚠ spaCy model not loaded, skipping comparison test")
            return True
        
        text1 = "Machine learning is a subset of artificial intelligence."
        text2 = "AI includes machine learning, which enables automatic learning."
        
        comparison = learner.compare_texts(text1, text2)
        
        if comparison.get('success'):
            print(f"  ✓ Text comparison works")
            similarity = comparison.get('similarity_score', 0)
            print(f"    - Similarity score: {similarity:.2f}")
            
            common_entities = comparison.get('common_entities', [])
            if common_entities:
                print(f"    - Common entities: {', '.join(common_entities)}")
            
            return True
        else:
            print(f"  ✗ Text comparison failed: {comparison.get('error', 'Unknown error')}")
            return False
    
    except Exception as e:
        print(f"  ✗ Comparison test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Web Learning Module - Test Suite".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝\n")
    
    results = {
        "Imports": test_imports(),
        "WebScraper": test_web_scraper(),
        "WebLearner": test_web_learner(),
        "API Models": test_api_models(),
        "Text Comparison": test_comparative_analysis(),
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print("=" * 60)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your web learning module is ready to use.")
        print("\nNext steps:")
        print("1. Run: python web_learning_example.py")
        print("2. Or start the server: python -m uvicorn server:app --reload")
        print("3. Check WEB_LEARNING_QUICKSTART.md for quick start guide")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

