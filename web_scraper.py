"""
Web Scraper Module
Extracts text and structured information from websites using BeautifulSoup
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    """Scrapes and extracts information from web pages"""
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        """
        Initialize web scraper
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Number of retries for failed requests
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch_url(self, url: str) -> Optional[str]:
        """
        Fetch content from URL with retry logic
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching URL: {url} (Attempt {attempt + 1}/{self.max_retries})")
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
                    return None
        return None
    
    def extract_text(self, html: str) -> str:
        """
        Extract clean text from HTML
        
        Args:
            html: HTML content
            
        Returns:
            Extracted text
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""
    
    def extract_structured_data(self, html: str, url: str) -> Dict[str, Any]:
        """
        Extract structured information from HTML
        
        Args:
            html: HTML content
            url: Source URL
            
        Returns:
            Dictionary with structured data
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else "No title"
            
            # Extract meta description
            meta_desc = ""
            if soup.find("meta", attrs={"name": "description"}):
                meta_desc = soup.find("meta", attrs={"name": "description"}).get("content", "")
            
            # Extract headings
            headings = []
            for h in soup.find_all(['h1', 'h2', 'h3']):
                headings.append({
                    'level': h.name,
                    'text': h.get_text().strip()
                })
            
            # Extract paragraphs
            paragraphs = []
            for p in soup.find_all('p')[:10]:  # Limit to first 10 paragraphs
                text = p.get_text().strip()
                if len(text) > 20:  # Only include substantial paragraphs
                    paragraphs.append(text)
            
            # Extract links
            links = []
            for a in soup.find_all('a', href=True)[:20]:  # Limit to first 20 links
                link_url = urljoin(url, a['href'])
                link_text = a.get_text().strip()
                if link_text and link_text != '#':
                    links.append({
                        'text': link_text,
                        'url': link_url
                    })
            
            # Extract tables if present
            tables = []
            for table in soup.find_all('table')[:5]:  # Limit to first 5 tables
                table_data = []
                for tr in table.find_all('tr')[:10]:  # Limit rows
                    row = []
                    for td in tr.find_all(['td', 'th']):
                        row.append(td.get_text().strip())
                    if row:
                        table_data.append(row)
                if table_data:
                    tables.append(table_data)
            
            return {
                'url': url,
                'title': title,
                'meta_description': meta_desc,
                'headings': headings,
                'paragraphs': paragraphs,
                'links': links,
                'tables': tables,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error extracting structured data: {str(e)}")
            return {
                'url': url,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def scrape_url(self, url: str, extract_text: bool = True, 
                   extract_structured: bool = True) -> Dict[str, Any]:
        """
        Scrape a URL and extract information
        
        Args:
            url: URL to scrape
            extract_text: Whether to extract plain text
            extract_structured: Whether to extract structured data
            
        Returns:
            Dictionary with scraped data
        """
        html = self.fetch_url(url)
        if not html:
            return {
                'success': False,
                'url': url,
                'error': 'Failed to fetch URL',
                'timestamp': datetime.now().isoformat()
            }
        
        result = {
            'success': True,
            'url': url,
            'timestamp': datetime.now().isoformat()
        }
        
        if extract_text:
            result['text'] = self.extract_text(html)
        
        if extract_structured:
            result['structured'] = self.extract_structured_data(html, url)
        
        return result
    
    def scrape_multiple_urls(self, urls: List[str], 
                           extract_text: bool = True,
                           extract_structured: bool = True) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs
        
        Args:
            urls: List of URLs to scrape
            extract_text: Whether to extract plain text
            extract_structured: Whether to extract structured data
            
        Returns:
            List of scraping results
        """
        results = []
        for url in urls:
            try:
                result = self.scrape_url(url, extract_text, extract_structured)
                results.append(result)
            except Exception as e:
                logger.error(f"Error scraping {url}: {str(e)}")
                results.append({
                    'success': False,
                    'url': url,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results

