"""
OpenAI with Sources and Citations - Latest 2025 Approach (Fixed)
Uses the new Responses API with built-in web search and file search tools
"""

import os
from openai import OpenAI
from typing import List, Dict, Any, Optional
import json

class OpenAIWithSources2025:
    def __init__(self, api_key: str):
        """Initialize OpenAI client with the new Responses API"""
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"  # Can also use gpt-4.1 or o-series models
    
    def ask_with_web_search(self, question: str, model: str = None) -> dict:
        """
        Ask OpenAI a question with automatic web search and source citations
        
        Args:
            question: The question to ask
            model: Model to use (defaults to gpt-4o)
            
        Returns:
            Dictionary with answer, sources, and citation metadata
        """
        
        model_to_use = model or self.model
        
        try:
            response = self.client.responses.create(
                model=model_to_use,
                input=question,
                tools=[{"type": "web_search"}]
            )
            
            result = {
                'answer': None,
                'question': question,
                'model_used': model_to_use,
                'web_search_calls': [],
                'citations': [],
                'sources': [],
                'raw_output': None
            }
            
            # First, try the simple approach
            if hasattr(response, 'output_text'):
                result['answer'] = response.output_text
            
            # Process the detailed response output for citations
            if hasattr(response, 'output') and response.output:
                result['raw_output'] = str(response.output)
                
                for item in response.output:
                    # Handle web search calls
                    if hasattr(item, 'type') and item.type == 'web_search_call':
                        result['web_search_calls'].append({
                            'id': getattr(item, 'id', ''),
                            'status': getattr(item, 'status', ''),
                            'type': item.type
                        })
                    
                    # Handle assistant messages with content
                    elif hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'content') and item.content:
                            for content_item in item.content:
                                # Extract the main text response - try multiple attributes
                                if hasattr(content_item, 'type') and content_item.type == 'output_text':
                                    text_content = None
                                    # Try different possible attributes
                                    for attr in ['text', 'content', 'value']:
                                        if hasattr(content_item, attr):
                                            text_content = getattr(content_item, attr)
                                            break
                                    
                                    if text_content and not result['answer']:
                                        result['answer'] = text_content
                                
                                # Extract citations and annotations
                                if hasattr(content_item, 'annotations'):
                                    for annotation in content_item.annotations:
                                        if hasattr(annotation, 'type') and annotation.type == 'url_citation':
                                            citation = {
                                                'title': getattr(annotation, 'title', ''),
                                                'url': getattr(annotation, 'url', ''),
                                                'type': 'url_citation'
                                            }
                                            result['citations'].append(citation)
                                            result['sources'].append(citation)
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'answer': None,
                'question': question,
                'model_used': model_to_use,
                'web_search_calls': [],
                'citations': [],
                'sources': []
            }
    
    def ask_with_web_search_simple(self, question: str, model: str = None) -> dict:
        """
        Simplified version that just gets the text response from web search
        
        Args:
            question: The question to ask
            model: Model to use (defaults to gpt-4o)
            
        Returns:
            Dictionary with answer and basic info
        """
        
        model_to_use = model or self.model
        
        try:
            response = self.client.responses.create(
                model=model_to_use,
                input=question,
                tools=[{"type": "web_search"}]
            )
            
            # Get the text response
            answer = None
            if hasattr(response, 'output_text'):
                answer = response.output_text
            
            # Extract any URLs that might be in the text
            import re
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', answer or '')
            
            return {
                'answer': answer,
                'question': question,
                'model_used': model_to_use,
                'urls_found': urls,
                'search_used': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'answer': None,
                'question': question,
                'model_used': model_to_use,
                'search_used': True
            }
    
    def create_vector_store_and_upload_files(self, file_paths: List[str], 
                                           store_name: str = "knowledge_base") -> str:
        """
        Create a vector store and upload files for file search
        
        Args:
            file_paths: List of file paths to upload
            store_name: Name for the vector store
            
        Returns:
            Vector store ID
        """
        try:
            # Create vector store
            vector_store = self.client.beta.vector_stores.create(
                name=store_name
            )
            
            # Upload files
            file_streams = []
            for file_path in file_paths:
                file_streams.append(open(file_path, "rb"))
            
            # Add files to vector store
            file_batch = self.client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id,
                files=file_streams
            )
            
            # Close file streams
            for stream in file_streams:
                stream.close()
            
            print(f"Vector store created with ID: {vector_store.id}")
            print(f"Status: {file_batch.status}")
            print(f"File counts: {file_batch.file_counts}")
            
            return vector_store.id
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None
    
    def ask_with_file_search(self, question: str, vector_store_id: str, 
                           model: str = None) -> dict:
        """
        Ask OpenAI a question with file search using vector store
        
        Args:
            question: The question to ask
            vector_store_id: ID of the vector store to search
            model: Model to use (defaults to gpt-4o)
            
        Returns:
            Dictionary with answer, file citations, and metadata
        """
        
        model_to_use = model or self.model
        
        try:
            response = self.client.responses.create(
                model=model_to_use,
                input=question,
                tools=[{
                    "type": "file_search",
                    "file_search": {
                        "vector_store_ids": [vector_store_id]
                    }
                }]
            )
            
            result = {
                'answer': None,
                'question': question,
                'model_used': model_to_use,
                'vector_store_id': vector_store_id,
                'file_search_calls': [],
                'file_citations': [],
                'sources': []
            }
            
            # Get the text response
            if hasattr(response, 'output_text'):
                result['answer'] = response.output_text
            
            # Try to extract file citations from the response
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'type') and item.type == 'file_search_call':
                        result['file_search_calls'].append({
                            'id': getattr(item, 'id', ''),
                            'status': getattr(item, 'status', ''),
                            'type': item.type
                        })
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'answer': None,
                'question': question,
                'model_used': model_to_use,
                'vector_store_id': vector_store_id,
                'file_search_calls': [],
                'file_citations': [],
                'sources': []
            }
    
    def ask_with_manual_sources(self, question: str, sources: List[Dict[str, str]], 
                              model: str = None) -> dict:
        """
        Ask OpenAI a question with manually provided sources (traditional approach)
        
        Args:
            question: The question to ask
            sources: List of source dictionaries with 'content' and 'source' keys
            model: Model to use (defaults to gpt-4o)
            
        Returns:
            Dictionary with answer and source citations
        """
        
        model_to_use = model or self.model
        
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
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return {
                'answer': response.choices[0].message.content,
                'question': question,
                'model_used': model_to_use,
                'manual_sources': sources,
                'prompt_used': prompt,
                'usage': response.usage.model_dump() if response.usage else None
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'answer': None,
                'question': question,
                'model_used': model_to_use,
                'manual_sources': sources
            }
    
    def debug_response_structure(self, question: str) -> dict:
        """
        Debug function to see the actual structure of the response
        """
        try:
            response = self.client.responses.create(
                model=self.model,
                input=question,
                tools=[{"type": "web_search"}]
            )
            
            debug_info = {
                'response_type': type(response).__name__,
                'response_attributes': dir(response),
                'has_output': hasattr(response, 'output'),
                'has_output_text': hasattr(response, 'output_text'),
                'output_type': type(response.output).__name__ if hasattr(response, 'output') else None,
            }
            
            if hasattr(response, 'output_text'):
                debug_info['output_text'] = response.output_text
            
            if hasattr(response, 'output') and response.output:
                debug_info['output_items'] = []
                for i, item in enumerate(response.output):
                    item_info = {
                        'index': i,
                        'type': getattr(item, 'type', 'unknown'),
                        'attributes': dir(item),
                        'item_class': type(item).__name__
                    }
                    
                    if hasattr(item, 'content') and item.content:
                        item_info['content_items'] = []
                        for j, content_item in enumerate(item.content):
                            content_info = {
                                'content_index': j,
                                'content_type': getattr(content_item, 'type', 'unknown'),
                                'content_attributes': dir(content_item),
                                'content_class': type(content_item).__name__
                            }
                            
                            # Try to get text content
                            for attr in ['text', 'content', 'value']:
                                if hasattr(content_item, attr):
                                    content_info[f'has_{attr}'] = True
                                    content_info[f'{attr}_value'] = str(getattr(content_item, attr))[:200] + "..."
                            
                            item_info['content_items'].append(content_info)
                    
                    debug_info['output_items'].append(item_info)
            
            return debug_info
            
        except Exception as e:
            return {'error': str(e)}

def demonstrate_usage():
    """Demonstrate how to use the latest OpenAI API with sources"""
    
    # You need to set your API key
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    # Initialize the client
    openai_client = OpenAIWithSources2025(api_key)
    
    print("=== OpenAI Responses API with Sources (2025) ===\n")
    
    # Example 1: Simple web search
    question1 = "What are the latest developments in AI safety in 2025?"
    
    print(f"Question: {question1}")
    print("Using automatic web search (simple)...")
    
    result1 = openai_client.ask_with_web_search_simple(question1)
    
    if 'error' in result1:
        print(f"Error: {result1['error']}")
    else:
        print(f"Answer: {result1['answer']}\n")
        
        if result1.get('urls_found'):
            print("URLs found in response:")
            for url in result1['urls_found']:
                print(f"  - {url}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Debug the response structure
    print("Debugging response structure...")
    debug_info = openai_client.debug_response_structure("What is machine learning?")
    #print("Debug info:")
    #print(json.dumps(debug_info, indent=2, default=str))
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Manual sources (traditional approach that definitely works)
    manual_sources = [
        {
            'source': 'AI Safety Research Institute',
            'content': 'Recent advances in AI safety include improved alignment techniques, better interpretability methods, and robust evaluation frameworks for large language models.'
        },
        {
            'source': 'Stanford AI Lab',
            'content': 'Constitutional AI and RLHF have shown promising results in making AI systems more helpful, harmless, and honest in 2025.'
        }
    ]
    
    question2 = "What are the recent advances in AI safety?"
    
    print(f"Question: {question2}")
    print("Using manual sources (guaranteed to work)...")
    
    result2 = openai_client.ask_with_manual_sources(question2, manual_sources)
    
    if 'error' in result2:
        print(f"Error: {result2['error']}")
    else:
        print(f"Answer: {result2['answer']}\n")
        print("Manual Sources Used:")
        for i, source in enumerate(manual_sources):
            print(f"  {i+1}. {source['source']}")
            print(f"     Content: {source['content'][:100]}...\n")

def installation_guide():
    """Print installation instructions for the latest OpenAI setup"""
    
    print("=== Installation Guide for OpenAI with Sources (2025) ===\n")
    
    print("1. Install the OpenAI Python library:")
    print("   pip install openai")
    print()
    
    print("2. Get your API key from OpenAI:")
    print("   - Visit: https://platform.openai.com/")
    print("   - Sign in to your account")
    print("   - Go to API Keys section")
    print("   - Create a new API key")
    print()
    
    print("3. Set your API key as environment variable:")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    print()
    
    print("4. Key Features in 2025:")
    print("   ✓ New Responses API (replaces Chat Completions for agents)")
    print("   ✓ Built-in web search tool")
    print("   ✓ Built-in file search tool")
    print("   ✓ Automatic URL citations from web search")
    print("   ✓ File citations from document search")
    print("   ✓ Support for vector stores and metadata filtering")
    print("   ✓ Works with GPT-4o, GPT-4.1, and o-series models")
    print()
    
    print("5. Pricing (2025):")
    print("   - Web search: Standard token pricing")
    print("   - File search: $0.10/GB vector storage per day + $2.50/1k tool calls")
    print("   - Vector stores: $0.10/GB/day storage cost")
    print()
    
    print("6. Important Notes:")
    print("   - Responses API is new and may have evolving structure")
    print("   - Chat Completions API still supported for simple use cases")
    print("   - Assistants API will be deprecated by mid-2026")
    print("   - Built-in tools only work with Responses API, not Chat Completions")
    print("   - If web search citations don't work, use manual sources approach")

if __name__ == "__main__":
    #installation_guide()
    #print("\n" + "="*60 + "\n")
    
    # Uncomment to run the demonstration (requires API key)
    demonstrate_usage()