"""
Perplexity with Sources and Citations - Latest 2025 Approach
Uses Perplexity's Sonar models with built-in real-time search and automatic citations
"""

import os
import requests
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

class PerplexityWithSources2025:
    def __init__(self, api_key: str):
        """Initialize Perplexity client with API key"""
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # Default to sonar-pro for best search capabilities
        self.default_model = "sonar-pro"
    
    def ask_with_web_search(self, question: str, model: str = None, 
                          search_options: Dict[str, Any] = None) -> dict:
        """
        Ask Perplexity a question with automatic web search and citations
        
        Args:
            question: The question to ask
            model: Model to use (sonar-pro, sonar, sonar-deep-research)
            search_options: Optional search configuration
            
        Returns:
            Dictionary with answer, sources, citations, and metadata
        """
        
        model_to_use = model or self.default_model
        
        # Prepare the request payload
        payload = {
            "model": model_to_use,
            "messages": [
                {
                    "role": "system",
                    "content": "Be precise and provide comprehensive answers with proper citations."
                },
                {
                    "role": "user", 
                    "content": question
                }
            ]
        }
        
        # Add search options if provided
        if search_options:
            payload.update(search_options)
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            data = response.json()
            
            result = {
                'answer': None,
                'question': question,
                'model_used': model_to_use,
                'search_results': [],
                'citations': [],
                'usage': {},
                'raw_response': data
            }
            
            # Extract the main response
            if 'choices' in data and data['choices']:
                choice = data['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    result['answer'] = choice['message']['content']
            
            # Extract search results (new format)
            if 'search_results' in data:
                result['search_results'] = data['search_results']
                # Convert to citations format for consistency
                result['citations'] = [
                    {
                        'title': sr.get('title', ''),
                        'url': sr.get('url', ''),
                        'date': sr.get('date', ''),
                        'type': 'web_search'
                    }
                    for sr in data['search_results']
                ]
            
            # Extract usage information
            if 'usage' in data:
                result['usage'] = data['usage']
            
            return result
            
        except requests.exceptions.RequestException as e:
            return {
                'error': f"API request failed: {str(e)}",
                'answer': None,
                'question': question,
                'model_used': model_to_use
            }
        except Exception as e:
            return {
                'error': f"Unexpected error: {str(e)}",
                'answer': None,
                'question': question,
                'model_used': model_to_use
            }
    
    def ask_with_search_filters(self, question: str, 
                              domain_filter: List[str] = None,
                              recency_filter: str = None,
                              academic_mode: bool = False,
                              model: str = None) -> dict:
        """
        Ask Perplexity with advanced search filtering options
        
        Args:
            question: The question to ask
            domain_filter: List of domains to include/exclude (prefix with - to exclude)
            recency_filter: Time filter ('month', 'week', 'day', 'hour')
            academic_mode: Use academic filter for scholarly sources
            model: Model to use
            
        Returns:
            Dictionary with filtered search results and citations
        """
        
        search_options = {}
        
        if domain_filter:
            search_options['search_domain_filter'] = domain_filter
        
        if recency_filter:
            search_options['search_recency_filter'] = recency_filter
        
        if academic_mode:
            search_options['search_mode'] = 'academic'
        
        # Add return related questions for better exploration
        search_options['return_related_questions'] = True
        
        return self.ask_with_web_search(question, model, search_options)
    
    def ask_with_date_range(self, question: str, 
                          after_date: str = None, 
                          before_date: str = None,
                          model: str = None) -> dict:
        """
        Ask Perplexity with date range filtering
        
        Args:
            question: The question to ask
            after_date: Start date filter (MM/DD/YYYY format)
            before_date: End date filter (MM/DD/YYYY format)
            model: Model to use
            
        Returns:
            Dictionary with date-filtered results
        """
        
        search_options = {}
        
        if after_date:
            search_options['search_after_date_filter'] = after_date
        
        if before_date:
            search_options['search_before_date_filter'] = before_date
        
        return self.ask_with_web_search(question, model, search_options)
    
    def ask_with_location(self, question: str, 
                        latitude: float, longitude: float,
                        country: str = None,
                        model: str = None) -> dict:
        """
        Ask Perplexity with location-based search filtering
        
        Args:
            question: The question to ask
            latitude: User latitude
            longitude: User longitude
            country: Optional ISO country code
            model: Model to use
            
        Returns:
            Dictionary with location-filtered results
        """
        
        search_options = {
            'web_search_options': {
                'user_location': {
                    'latitude': latitude,
                    'longitude': longitude
                }
            }
        }
        
        if country:
            search_options['web_search_options']['user_location']['country'] = country
        
        return self.ask_with_web_search(question, model, search_options)
    
    def ask_with_structured_output(self, question: str, 
                                 output_schema: Dict[str, Any],
                                 model: str = None) -> dict:
        """
        Ask Perplexity with structured JSON output
        
        Args:
            question: The question to ask
            output_schema: JSON schema for structured output
            model: Model to use
            
        Returns:
            Dictionary with structured response
        """
        
        search_options = {
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'schema': output_schema
                }
            }
        }
        
        return self.ask_with_web_search(question, model, search_options)
    
    def ask_research_question(self, question: str, 
                            research_domains: List[str] = None) -> dict:
        """
        Ask a research question using sonar-deep-research model
        
        Args:
            question: The research question
            research_domains: List of academic/research domains to focus on
            
        Returns:
            Dictionary with comprehensive research results
        """
        
        # Use deep research model for comprehensive analysis
        model = "sonar-deep-research"
        
        # Default to academic sources if no domains specified
        if not research_domains:
            research_domains = [
                'arxiv.org', 'researchgate.net', 'scholar.google.com',
                'pubmed.ncbi.nlm.nih.gov', 'ieee.org', 'nature.com',
                'sciencedirect.com', 'springer.com'
            ]
        
        search_options = {
            'search_domain_filter': research_domains,
            'web_search_options': {
                'search_context_size': 'high'
            },
            'return_related_questions': True
        }
        
        return self.ask_with_web_search(question, model, search_options)
    
    def format_response_with_citations(self, result: dict) -> str:
        """
        Format a Perplexity response with citations in a readable way
        
        Args:
            result: Result dictionary from any ask method
            
        Returns:
            Formatted string with answer and citations
        """
        
        if 'error' in result:
            return f"Error: {result['error']}"
        
        formatted_response = f"**Question:** {result['question']}\n\n"
        formatted_response += f"**Answer:**\n{result['answer']}\n\n"
        
        if result.get('search_results') or result.get('citations'):
            sources = result.get('search_results') or result.get('citations')
            formatted_response += "**Sources:**\n"
            
            for i, source in enumerate(sources, 1):
                formatted_response += f"{i}. **{source.get('title', 'Unknown Title')}**\n"
                formatted_response += f"   URL: {source.get('url', 'No URL')}\n"
                
                if source.get('date'):
                    formatted_response += f"   Date: {source['date']}\n"
                
                formatted_response += "\n"
        
        if result.get('usage'):
            usage = result['usage']
            formatted_response += f"**Usage:** {usage}\n"
        
        return formatted_response

def demonstrate_usage():
    """Demonstrate how to use Perplexity with sources and citations"""
    
    # You need to set your API key
    api_key = os.getenv('PERPLEXITY_API_KEY')
    
    if not api_key:
        print("Please set your PERPLEXITY_API_KEY environment variable")
        return
    
    # Initialize the client
    perplexity_client = PerplexityWithSources2025(api_key)
    
    print("=== Perplexity API with Real-time Search and Citations (2025) ===\n")
    
    # Example 1: Basic web search with citations
    question1 = "What are the latest developments in quantum computing in 2025?"
    
    print(f"Question: {question1}")
    print("Using Perplexity's real-time web search...")
    
    result1 = perplexity_client.ask_with_web_search(question1)
    
    if 'error' in result1:
        print(f"Error: {result1['error']}")
    else:
        print("Response:")
        print(perplexity_client.format_response_with_citations(result1))
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Search with domain filtering
    question2 = "What are the latest Python developments?"
    
    print(f"Question: {question2}")
    print("Using domain filtering for tech sources...")
    
    tech_domains = ['github.com', 'stackoverflow.com', 'python.org', 'pypi.org']
    result2 = perplexity_client.ask_with_search_filters(
        question2, 
        domain_filter=tech_domains,
        recency_filter='week'
    )
    
    if 'error' in result2:
        print(f"Error: {result2['error']}")
    else:
        print("Response:")
        print(perplexity_client.format_response_with_citations(result2))
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Academic research
    question3 = "Recent advances in machine learning for climate modeling"
    
    print(f"Question: {question3}")
    print("Using academic research mode...")
    
    result3 = perplexity_client.ask_research_question(question3)
    
    if 'error' in result3:
        print(f"Error: {result3['error']}")
    else:
        print("Research Response:")
        print(perplexity_client.format_response_with_citations(result3))
    
    print("\n" + "="*60 + "\n")
    
    # Example 4: Structured output
    question4 = "Compare renewable energy adoption in the top 5 countries"
    
    print(f"Question: {question4}")
    print("Using structured JSON output...")
    
    schema = {
        "type": "object",
        "properties": {
            "countries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "renewable_percentage": {"type": "number"},
                        "primary_sources": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "summary": {"type": "string"}
        }
    }
    
    result4 = perplexity_client.ask_with_structured_output(question4, schema)
    
    if 'error' in result4:
        print(f"Error: {result4['error']}")
    else:
        print("Structured Response:")
        print(result4['answer'])
        print("\nSources:")
        for source in result4.get('citations', []):
            print(f"- {source['title']}: {source['url']}")

def installation_guide():
    """Print installation and setup instructions for Perplexity API"""
    
    print("=== Installation Guide for Perplexity API with Sources (2025) ===\n")
    
    print("1. Install required libraries:")
    print("   pip install requests")
    print()
    
    print("2. Get your API key from Perplexity:")
    print("   - Visit: https://www.perplexity.ai/")
    print("   - Go to Settings > API")
    print("   - Add payment method and purchase credits")
    print("   - Generate API key")
    print("   - Pro subscribers get $5 monthly credits")
    print()
    
    print("3. Set your API key as environment variable:")
    print("   export PERPLEXITY_API_KEY='your-api-key-here'")
    print()
    
    print("4. Key Features in 2025:")
    print("   ✓ Real-time web search with every query")
    print("   ✓ Automatic citations included by default")
    print("   ✓ Multiple Sonar models (pro, standard, deep-research)")
    print("   ✓ Search filtering (domain, date, location, academic)")
    print("   ✓ Structured JSON output support")
    print("   ✓ OpenAI-compatible API format")
    print("   ✓ Citations no longer cost extra tokens")
    print()
    
    print("5. Available Models:")
    print("   - sonar-pro: Advanced search with comprehensive answers")
    print("   - sonar: Standard search capabilities")
    print("   - sonar-deep-research: Comprehensive research analysis")
    print("   - Plus open-source models (Llama, Mistral, etc.)")
    print()
    
    print("6. Search Capabilities:")
    print("   ✓ Domain filtering (include/exclude specific sites)")
    print("   ✓ Date range filtering (MM/DD/YYYY format)")
    print("   ✓ Recency filtering (hour, day, week, month)")
    print("   ✓ Academic mode for scholarly sources")
    print("   ✓ Location-based search results")
    print("   ✓ Related questions suggestions")
    print()
    
    print("7. Pricing Notes:")
    print("   - Pay-per-use model with credits")
    print("   - Citations included at no extra cost")
    print("   - Pro subscribers get $5 monthly credits")
    print("   - Variable pricing based on model and usage")
    print()
    
    print("8. Important Notes:")
    print("   - All responses include citations by default")
    print("   - Real-time web search with every query")
    print("   - OpenAI-compatible API format for easy integration")
    print("   - Rate limit: 50 requests/min for Sonar models")
    print("   - search_results field replaces deprecated citations field")
    print()
    
    print("9. Best Use Cases:")
    print("   - Real-time information retrieval")
    print("   - Research with academic sources")
    print("   - Current events and news analysis")
    print("   - Technical documentation searches")
    print("   - Location-specific information")
    print("   - Structured data extraction from web sources")

if __name__ == "__main__":
    #installation_guide()
    #print("\n" + "="*60 + "\n")
    
    # Uncomment to run the demonstration (requires API key)
    demonstrate_usage()