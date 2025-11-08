"""
Web Learner Module
Uses spaCy for NLP processing and learning from web content
"""

import spacy
from typing import Dict, List, Any, Optional, Set
import json
import logging
from datetime import datetime
from collections import Counter
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebLearner:
    """Learn from web content using NLP"""
    
    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize web learner with spaCy model
        
        Args:
            model: spaCy model to use
        """
        try:
            self.nlp = spacy.load(model)
            logger.info(f"Loaded spaCy model: {model}")
        except OSError:
            logger.warning(f"Model {model} not found. Download it with: python -m spacy download {model}")
            self.nlp = None
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities from text
        
        Args:
            text: Text to process
            
        Returns:
            List of entities with types
        """
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text[:1000000])  # Limit text size for performance
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text
        
        Args:
            text: Text to process
            
        Returns:
            List of noun phrases
        """
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text[:1000000])
            noun_phrases = []
            
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 4:  # Limit phrase length
                    noun_phrases.append(chunk.text.lower())
            
            return noun_phrases
        except Exception as e:
            logger.error(f"Error extracting noun phrases: {str(e)}")
            return []
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Extract keywords from text using TF-IDF-like approach
        
        Args:
            text: Text to process
            top_n: Number of top keywords to return
            
        Returns:
            List of keywords with scores
        """
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text[:1000000])
            
            # Extract tokens with high information content
            keywords = []
            word_freq = Counter()
            
            for token in doc:
                # Filter for meaningful tokens
                if (not token.is_stop and 
                    not token.is_punct and 
                    token.is_alpha and
                    len(token.text) > 2):
                    word_freq[token.text.lower()] += 1
            
            # Get top keywords
            for word, freq in word_freq.most_common(top_n):
                keywords.append({
                    'keyword': word,
                    'frequency': freq,
                    'score': freq / max(word_freq.values()) if word_freq.values() else 0
                })
            
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def sentiment_analysis(self, text: str) -> Dict[str, float]:
        """
        Simple sentiment analysis based on token attributes
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment metrics
        """
        if not self.nlp:
            return {'sentiment': 'unknown', 'score': 0}
        
        try:
            doc = self.nlp(text[:100000])  # Limit for sentiment
            
            # Simple approach: count positive/negative word vectors
            positive_tokens = []
            negative_tokens = []
            
            for token in doc:
                if token.has_vector and not token.is_stop:
                    if token.vector_norm > 0:
                        positive_tokens.append(token)
            
            # For more robust sentiment, consider using a transformer model
            # This is a basic placeholder
            sentiment_score = 0
            if positive_tokens:
                sentiment_score = min(len(positive_tokens) / len(doc), 1.0)
            
            return {
                'sentiment': 'positive' if sentiment_score > 0.6 else 'negative' if sentiment_score < 0.4 else 'neutral',
                'score': sentiment_score,
                'token_count': len(doc)
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {'sentiment': 'unknown', 'score': 0}
    
    def summarize_text(self, text: str, num_sentences: int = 3) -> List[str]:
        """
        Extract key sentences for summary using sentence importance
        
        Args:
            text: Text to summarize
            num_sentences: Number of sentences to extract
            
        Returns:
            List of important sentences
        """
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text[:200000])  # Limit text size
            
            sentences = list(doc.sents)
            if len(sentences) <= num_sentences:
                return [sent.text for sent in sentences]
            
            # Score sentences based on word frequency
            word_freq = Counter()
            for token in doc:
                if not token.is_stop and not token.is_punct and token.is_alpha:
                    word_freq[token.text.lower()] += 1
            
            # Score each sentence
            sentence_scores = {}
            for i, sent in enumerate(sentences):
                score = 0
                for token in sent:
                    if token.text.lower() in word_freq:
                        score += word_freq[token.text.lower()]
                sentence_scores[i] = score / len(sent) if len(sent) > 0 else 0
            
            # Get top sentences, preserving order
            top_indices = sorted(
                sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences],
                key=lambda x: x[0]
            )
            
            summary = [sentences[i].text for i, _ in top_indices]
            return summary
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return []
    
    def analyze_content(self, text: str, url: str = "") -> Dict[str, Any]:
        """
        Comprehensive analysis of web content
        
        Args:
            text: Content text to analyze
            url: Source URL for reference
            
        Returns:
            Dictionary with complete analysis
        """
        if not self.nlp:
            return {
                'success': False,
                'error': 'spaCy model not loaded'
            }
        
        try:
            analysis = {
                'url': url,
                'timestamp': datetime.now().isoformat(),
                'text_length': len(text),
                'word_count': len(text.split()),
                'entities': self.extract_entities(text),
                'keywords': self.extract_keywords(text),
                'noun_phrases': self.extract_noun_phrases(text),
                'sentiment': self.sentiment_analysis(text),
                'summary': self.summarize_text(text, num_sentences=5),
                'success': True
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Error in content analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url,
                'timestamp': datetime.now().isoformat()
            }
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare two texts and find similarities
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Comparison results
        """
        if not self.nlp:
            return {'similarity': 0, 'error': 'spaCy model not loaded'}
        
        try:
            doc1 = self.nlp(text1[:100000])
            doc2 = self.nlp(text2[:100000])
            
            # Calculate similarity
            similarity = doc1.similarity(doc2)
            
            # Extract common entities
            entities1 = {ent.text for ent in doc1.ents}
            entities2 = {ent.text for ent in doc2.ents}
            common_entities = entities1.intersection(entities2)
            
            # Extract common noun phrases
            phrases1 = set(self.extract_noun_phrases(text1))
            phrases2 = set(self.extract_noun_phrases(text2))
            common_phrases = phrases1.intersection(phrases2)
            
            return {
                'similarity_score': float(similarity),
                'common_entities': list(common_entities),
                'common_phrases': list(common_phrases),
                'unique_entities_text1': list(entities1 - entities2),
                'unique_entities_text2': list(entities2 - entities1),
                'success': True
            }
        except Exception as e:
            logger.error(f"Error comparing texts: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def extract_learning_data(self, text: str, url: str = "") -> Dict[str, Any]:
        """
        Extract structured learning data from text
        
        Args:
            text: Text to extract learning data from
            url: Source URL
            
        Returns:
            Structured learning data
        """
        if not self.nlp:
            return {'success': False, 'error': 'spaCy model not loaded'}
        
        try:
            doc = self.nlp(text[:500000])
            
            # Extract different types of information
            definitions = []
            facts = []
            questions = []
            
            # Simple heuristic: sentences with "is" are likely definitions
            # Sentences starting with common question words are questions
            for sent in doc.sents:
                sent_text = sent.text.strip()
                
                if ' is ' in sent_text and len(sent_text) < 200:
                    definitions.append(sent_text)
                
                if sent_text.startswith(('What', 'Where', 'When', 'Why', 'How')):
                    questions.append(sent_text)
                
                # Any significant sentence is a potential fact
                if len(sent_text.split()) >= 5 and len(sent_text) < 300:
                    facts.append(sent_text)
            
            return {
                'url': url,
                'definitions': definitions[:10],
                'facts': facts[:20],
                'questions': questions[:10],
                'entities': self.extract_entities(text),
                'key_terms': [kw['keyword'] for kw in self.extract_keywords(text, top_n=10)],
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
        except Exception as e:
            logger.error(f"Error extracting learning data: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url,
                'timestamp': datetime.now().isoformat()
            }

