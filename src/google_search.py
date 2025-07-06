"""
Google Search API tool for the QnA Agent.

This module implements a search tool that uses Google's Custom Search API
to find relevant web content and extract information.
"""

import os
from typing import Optional, Dict, Any
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from bs4 import BeautifulSoup


class GoogleSearchTool:
    """
    A tool for searching Google and extracting content from web pages.
    
    This class provides an interface for Google Custom Search API operations,
    with built-in error handling and retry logic for reliability.
    """
    
    def __init__(self):
        """Initialize the Google Search tool using environment variables."""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        if not self.search_engine_id:
            raise ValueError("Google Search Engine ID is required. Set GOOGLE_SEARCH_ENGINE_ID environment variable.")
        
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def search_and_extract(self, query: str, max_length: Optional[int] = 5000) -> Dict[str, Any]:
        """
        Search Google and extract content from the top result.
        
        Args:
            query: The search query
            max_length: Maximum character length of content to return
            
        Returns:
            Dictionary with 'title', 'url', and 'content' of the top result
            
        Raises:
            ValueError: If no search results are found
        """
        # Search Google
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': 1  # We only need the top result
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'items' not in data or not data['items']:
            raise ValueError(f"No Google search results found for query: {query}")
        
        # Get the top result
        top_result = data['items'][0]
        result = {
            'title': top_result.get('title', ''),
            'url': top_result.get('link', ''),
        }
        
        # Extract content from the URL
        try:
            content = self._extract_content(result['url'], max_length)
        except Exception:
            # Fallback to snippet if content extraction fails
            content = top_result.get('snippet', 'Content extraction failed')
        
        result['content'] = content
        return result
    
    def _extract_content(self, url: str, max_length: Optional[int]) -> str:
        """
        Extract and clean text content from a web page.
        
        Args:
            url: The URL of the web page
            max_length: Maximum character length of content to return
            
        Returns:
            The cleaned text content
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text - this is the good text cleaning logic we're keeping
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit length if specified
        if max_length and len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text