"""
Automated Learning Pipeline
Searches the web, learns from content, and saves knowledge to knowledge base
NO image generation or model training here
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from web_scraper import WebScraper
from web_learner import WebLearner
from model_persistence import KnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoTrainer:
    """Automatically learn from web and save to knowledge base"""
    
    def __init__(self):
        """Initialize auto trainer"""
        self.scraper = WebScraper()
        self.learner = WebLearner()
        self.kb = KnowledgeBase()
        self.training_log = []
    
    def search_urls_for_topic(self, topic: str, count: int = 5) -> List[str]:
        topic_encoded = topic.replace(' ', '+')
        topic_slug = topic.replace(' ', '-').lower()
        """
        Generate search URLs for a topic
        
        Args:
            topic: Topic to search for
            count: Number of URLs to generate
            
        Returns:
            List of URLs to scrape
        """
        urls = [
                f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                f"https://www.coursera.org/search?query={topic_encoded}",
                f"https://www.kaggle.com/search?q={topic_encoded}",
                f"https://towardsdatascience.com/tagged/{topic_slug}",
                f"https://www.datacamp.com/blog/how-to-learn-{topic_slug}",
                f"https://roadmap.sh/{topic_slug}",
                f"https://www.analyticsvidhya.com/blog/tag/{topic_slug}"
            ]
        
        return urls[:count]
    
    def learn_from_topic(self, topic: str, urls: Optional[List[str]] = None, 
                        max_urls: int = 5) -> Dict[str, Any]:
        """
        Search web and learn from content for a topic
        
        Args:
            topic: Topic to learn about
            urls: Optional custom list of URLs
            max_urls: Maximum URLs to scrape
            
        Returns:
            Dictionary with learning data
        """
        logger.info("=" * 60)
        logger.info(f"LEARNING FROM WEB: {topic}")
        logger.info("=" * 60)
        
        # Generate URLs if not provided
        if urls is None:
            urls = self.search_urls_for_topic(topic, max_urls)
        else:
            urls = urls[:max_urls]
        
        all_texts = []
        all_entities = []
        all_keywords = []
        successful_urls = 0
        
        # Scrape each URL
        for i, url in enumerate(urls, 1):
            logger.info(f"\n[{i}/{len(urls)}] Scraping: {url}")
            
            try:
                result = self.scraper.scrape_url(url)
                
                if result['success']:
                    text = result['text']
                    all_texts.append(text)
                    successful_urls += 1
                    
                    # Extract learning data
                    if text:
                        entities = self.learner.extract_entities(text)
                        keywords = self.learner.extract_keywords(text, top_n=10)
                        
                        all_entities.extend(entities)
                        all_keywords.extend(keywords)
                        
                        logger.info(f"  ✓ Extracted {len(entities)} entities, {len(keywords)} keywords")
                else:
                    logger.warning(f"  ✗ Failed to scrape: {result.get('error')}")
            
            except Exception as e:
                logger.error(f"  ✗ Error scraping {url}: {str(e)}")
        
        # Combine all learned data
        combined_text = " ".join(all_texts)
        
        if combined_text:
            analysis = self.learner.analyze_content(combined_text, f"topic:{topic}")
            
            # Save to knowledge base
            self.kb.add_knowledge(topic, analysis)
            
            logger.info(f"\n✓ Learning complete for '{topic}'")
            logger.info(f"  URLs processed: {successful_urls}/{len(urls)}")
            logger.info(f"  Total text: {len(combined_text)} characters")
            logger.info(f"  Entities found: {len(all_entities)}")
            logger.info(f"  Keywords found: {len(all_keywords)}")
            
            return {
                'success': True,
                'topic': topic,
                'text_length': len(combined_text),
                'analysis': analysis,
                'entities_count': len(all_entities),
                'keywords_count': len(all_keywords),
                'urls_processed': successful_urls,
                'message': f'Successfully learned about {topic} from {successful_urls} sources'
            }
        else:
            logger.warning(f"No text extracted for topic: {topic}")
            return {
                'success': False,
                'topic': topic,
                'error': 'No text could be extracted from URLs',
                'urls_processed': successful_urls
            }
    
    def get_knowledge(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Get saved knowledge about a topic
        
        Args:
            topic: Topic name
            
        Returns:
            Learned knowledge or None
        """
        logger.info(f"Retrieving knowledge about: {topic}")
        knowledge = self.kb.get_knowledge(topic)
        
        if knowledge:
            logger.info(f"✓ Knowledge found for {topic}")
            logger.info(f"  Keywords: {len(knowledge.get('key_terms', []))}")
            logger.info(f"  Entities: {len(knowledge.get('entities', []))}")
        else:
            logger.info(f"✗ No knowledge found for {topic}")
        
        return knowledge
    
    def search_knowledge(self, keyword: str) -> list:
        """
        Search knowledge base for keyword
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            Search results
        """
        logger.info(f"Searching knowledge base for: {keyword}")
        results = self.kb.search_knowledge(keyword)
        logger.info(f"Found {len(results)} results")
        return results
    
    def list_topics(self) -> List[str]:
        """Get all topics in knowledge base"""
        topics = self.kb.list_topics()
        logger.info(f"Topics in knowledge base: {len(topics)}")
        for topic in topics:
            logger.info(f"  - {topic}")
        return topics


# Example usage
if __name__ == "__main__":
    trainer = AutoTrainer()
    
    # Learn about a topic
    result = trainer.learn_from_topic("Machine Learning")
    
    print("\n" + "=" * 60)
    print("LEARNING COMPLETE!")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))
    
    # Get what we learned
    knowledge = trainer.get_knowledge("Machine Learning")
    if knowledge:
        print("\n" + "=" * 60)
        print("SAVED KNOWLEDGE")
        print("=" * 60)
        print(f"Key Terms: {knowledge.get('key_terms', [])[:10]}")
        print(f"Entities: {len(knowledge.get('entities', []))} found")
