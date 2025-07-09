"""
Gemini 2.0 with Sources and Citations - Latest 2025 Approach
Uses the new google-genai SDK (v1.0) with built-in Google Search grounding
"""

import os
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    Tool,
    ToolConfig
)

class GeminiWithSources2025:
    def __init__(self, api_key: str):
        """Initialize Gemini client with the new Google GenAI SDK"""
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"  # Latest model with search capabilities
    
    def ask_with_web_search(self, question: str, use_search: bool = True) -> dict:
        """
        Ask Gemini a question with automatic web search and source citations
        
        Args:
            question: The question to ask
            use_search: Whether to use Google Search grounding
            
        Returns:
            Dictionary with answer, sources, and citation metadata
        """
        
        if use_search:
            # Configure Google Search tool
            search_tool = Tool(google_search=GoogleSearch())
            
            config = GenerateContentConfig(
                tools=[search_tool],
                temperature=0.1,  # Lower temperature for more factual responses
                max_output_tokens=2048
            )
        else:
            config = GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048
            )
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=question,
                config=config
            )
            
            result = {
                'answer': response.text,
                'question': question,
                'used_search': use_search,
                'sources': [],
                'citations': [],
                'grounding_metadata': None
            }
            
            # Extract grounding metadata and sources if available
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Extract citation metadata
                if hasattr(candidate, 'citation_metadata') and candidate.citation_metadata:
                    citations = []
                    for citation in candidate.citation_metadata.citations:
                        citations.append({
                            'start_index': citation.start_index,
                            'end_index': citation.end_index,
                            'uri': citation.uri,
                            'title': getattr(citation, 'title', ''),
                            'license': getattr(citation, 'license', ''),
                            'publication_date': getattr(citation, 'publication_date', None)
                        })
                    result['citations'] = citations
                
                # Extract grounding metadata (web search sources)
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    grounding_chunks = []
                    grounding_supports = []
                    
                    # Extract grounding chunks (web sources)
                    if hasattr(candidate.grounding_metadata, 'grounding_chunks'):
                        for chunk in candidate.grounding_metadata.grounding_chunks:
                            if hasattr(chunk, 'web') and chunk.web:
                                grounding_chunks.append({
                                    'title': chunk.web.title,
                                    'uri': chunk.web.uri
                                })
                    
                    # Extract grounding supports (which parts of text are supported by which sources)
                    if hasattr(candidate.grounding_metadata, 'grounding_supports'):
                        for support in candidate.grounding_metadata.grounding_supports:
                            grounding_supports.append({
                                'grounding_chunk_indices': getattr(support, 'grounding_chunk_indices', []),
                                'confidence_scores': getattr(support, 'confidence_scores', []),
                                'segment_text': getattr(support.segment, 'text', '') if hasattr(support, 'segment') else '',
                                'start_index': getattr(support.segment, 'start_index', 0) if hasattr(support, 'segment') else 0,
                                'end_index': getattr(support.segment, 'end_index', 0) if hasattr(support, 'segment') else 0
                            })
                    
                    result['sources'] = grounding_chunks
                    result['grounding_supports'] = grounding_supports
                    result['grounding_metadata'] = {
                        'grounding_chunks': grounding_chunks,
                        'grounding_supports': grounding_supports
                    }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'answer': None,
                'question': question,
                'used_search': use_search,
                'sources': [],
                'citations': []
            }
    
    def ask_with_manual_sources(self, question: str, sources: list) -> dict:
        """
        Ask Gemini a question with manually provided sources
        
        Args:
            question: The question to ask
            sources: List of source dictionaries with 'content' and 'source' keys
            
        Returns:
            Dictionary with answer and source citations
        """
        
        # Build prompt with sources
        sources_text = "\n\n".join([
            f"Source {i+1} ({source.get('source', 'Unknown')}):\n{source.get('content', '')}"
            for i, source in enumerate(sources)
        ])
        
        prompt = f"""Based on the following sources, please answer the question and cite your sources using [Source X] notation.

Sources:
{sources_text}

Question: {question}

Please provide a comprehensive answer and cite specific sources for each claim using [Source X] format.
"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=GenerateContentConfig(temperature=0.1)
            )
            
            return {
                'answer': response.text,
                'question': question,
                'manual_sources': sources,
                'prompt_used': prompt,
                'used_search': False
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'answer': None,
                'question': question,
                'manual_sources': sources
            }

def demonstrate_usage():
    """Demonstrate how to use the latest Gemini API with sources"""
    
    # You need to set your API key
    api_key = os.getenv('GEMINI_API_KEY')  # Set this environment variable
    
    if not api_key:
        print("Please set your GEMINI_API_KEY environment variable")
        return
    
    # Initialize the client
    gemini_client = GeminiWithSources2025(api_key)
    
    print("=== Gemini 2.0 with Web Search Sources ===\n")
    
    # Example 1: Ask a question with automatic web search
    question1 = "What are the latest developments in quantum computing in 2025?"
    
    print(f"Question: {question1}")
    print("Using automatic web search...")
    
    result1 = gemini_client.ask_with_web_search(question1, use_search=True)
    
    if 'error' in result1:
        print(f"Error: {result1['error']}")
    else:
        print(f"Answer: {result1['answer']}\n")
        
        if result1['sources']:
            print("Web Sources Found:")
            for i, source in enumerate(result1['sources']):
                print(f"  {i+1}. {source['title']}")
                print(f"     URL: {source['uri']}\n")
        
        if result1['citations']:
            print("Citations:")
            for citation in result1['citations']:
                print(f"  - Text position {citation['start_index']}-{citation['end_index']}")
                print(f"    Source: {citation['uri']}")
                if citation['title']:
                    print(f"    Title: {citation['title']}")
                print()
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Ask a question with manual sources
    manual_sources = [
        {
            'source': 'MIT Technology Review',
            'content': 'Quantum computing has made significant advances in error correction and fault-tolerant quantum systems in 2025, with several companies achieving important milestones in quantum advantage.'
        },
        {
            'source': 'Nature Physics',
            'content': 'Recent breakthroughs in quantum computing include improved qubit coherence times and the development of more stable quantum processors.'
        }
    ]
    
    question2 = "What are the recent advances in quantum computing?"
    
    print(f"Question: {question2}")
    print("Using manual sources...")
    
    result2 = gemini_client.ask_with_manual_sources(question2, manual_sources)
    
    if 'error' in result2:
        print(f"Error: {result2['error']}")
    else:
        print(f"Answer: {result2['answer']}\n")
        print("Manual Sources Used:")
        for i, source in enumerate(manual_sources):
            print(f"  {i+1}. {source['source']}")
            print(f"     Content: {source['content'][:100]}...\n")

# Installation instructions
def installation_guide():
    """Print installation instructions for the latest setup"""
    
    print("=== Installation Guide for Gemini 2.0 with Sources ===\n")
    
    print("1. Install the new Google GenAI SDK:")
    print("   pip install google-genai")
    print()
    
    print("2. Get your API key from Google AI Studio:")
    print("   - Visit: https://ai.google.dev/")
    print("   - Sign in with your Google account")
    print("   - Create an API key")
    print("   - Note: You need to upgrade your plan to use Google Search grounding")
    print()
    
    print("3. Set your API key as environment variable:")
    print("   export GEMINI_API_KEY='your-api-key-here'")
    print()
    
    print("4. Key Features in 2025:")
    print("   ✓ Built-in Google Search grounding")
    print("   ✓ Automatic source citations")
    print("   ✓ Grounding metadata with confidence scores")
    print("   ✓ Citation metadata with publication dates")
    print("   ✓ Support for manual source input")
    print("   ✓ Gemini 2.0 Flash model (latest)")
    print()
    
    print("5. Important Notes:")
    print("   - The google-generativeai package is being phased out")
    print("   - google-genai (v1.0) is the new recommended SDK")
    print("   - Google Search grounding requires a paid plan")
    print("   - CitationMetadata provides automatic source attribution")

if __name__ == "__main__":
    #installation_guide()
    #print("\n" + "="*60 + "\n")
    
    # Uncomment to run the demonstration (requires API key)
    demonstrate_usage()