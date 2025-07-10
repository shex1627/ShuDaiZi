"""
Complete Reranker Testing Suite
Testing BGE-Reranker-Base, MS-MARCO-MiniLM-L6-v2, and Cohere Rerank
"""

import os
import time
import json
import pandas as pd
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Test data for reranker evaluation
TEST_QUERIES_AND_DOCS = [
    {
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "Deep learning uses neural networks with many layers to process complex patterns in data.",
            "Python is a popular programming language used for data science and web development.",
            "Machine learning algorithms can be supervised, unsupervised, or reinforcement learning.",
            "Cooking recipes often involve multiple steps and ingredients to create delicious meals.",
            "The weather today is sunny with a chance of rain in the afternoon.",
            "Artificial intelligence encompasses machine learning, natural language processing, and computer vision.",
            "Statistical models help understand relationships between variables in datasets.",
        ]
    },
    {
        "query": "How to cook pasta?",
        "documents": [
            "To cook pasta, bring water to boil, add salt, then add pasta and cook for 8-12 minutes.",
            "Machine learning requires large datasets and computational resources for training models.",
            "Boil water in a large pot, add pasta when water is bubbling, stir occasionally.",
            "Neural networks consist of interconnected nodes that process information layer by layer.",
            "Different pasta shapes require different cooking times - check package instructions.",
            "Data preprocessing is crucial for machine learning model performance.",
            "Season the boiling water with salt before adding pasta for better flavor.",
            "Cloud computing provides scalable infrastructure for large-scale data processing.",
        ]
    },
    {
        "query": "Python programming basics",
        "documents": [
            "Python is an interpreted, high-level programming language with dynamic semantics.",
            "Variables in Python don't need explicit declaration and can change types dynamically.",
            "Pasta should be cooked until al dente, which means firm to the bite.",
            "Python supports multiple programming paradigms including object-oriented and functional.",
            "The weather forecast shows temperatures will drop significantly next week.",
            "Indentation is crucial in Python as it defines code blocks instead of braces.",
            "Python has extensive libraries for data science, web development, and automation.",
            "Cooking involves understanding flavors, techniques, and timing for best results.",
        ]
    }
]

class BGEReranker:
    """BGE-Reranker-Base implementation"""
    
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the BGE reranker model"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            print(f"Loading BGE reranker: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            print("✓ BGE reranker loaded successfully")
            
        except ImportError as e:
            print(f"✗ Cannot load BGE reranker: {e}")
            print("Please install: pip install torch transformers")
            self.model = None
            self.tokenizer = None
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Dict]:
        """Rerank documents based on query relevance"""
        if self.model is None or self.tokenizer is None:
            return self._mock_rerank(query, documents, top_k)
        
        try:
            import torch
            
            # Prepare pairs
            pairs = [[query, doc] for doc in documents]
            
            # Tokenize
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt', 
                    max_length=512
                )
                
                # Get scores
                scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
                scores = scores.detach().cpu().numpy()
            
            # Create results with scores
            results = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                results.append({
                    'document': doc,
                    'score': float(score),
                    'index': i
                })
            
            # Sort by score (higher is better)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            if top_k:
                results = results[:top_k]
            
            return results
            
        except Exception as e:
            print(f"Error in BGE reranking: {e}")
            return self._mock_rerank(query, documents, top_k)
    
    def _mock_rerank(self, query: str, documents: List[str], top_k: int = None):
        """Mock reranking for when model is not available"""
        import random
        results = []
        for i, doc in enumerate(documents):
            # Simple keyword matching score + random for demo
            keywords = query.lower().split()
            doc_lower = doc.lower()
            keyword_score = sum(1 for kw in keywords if kw in doc_lower)
            mock_score = keyword_score + random.uniform(-0.5, 0.5)
            
            results.append({
                'document': doc,
                'score': mock_score,
                'index': i
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        if top_k:
            results = results[:top_k]
        return results


class MSMarcoReranker:
    """MS-MARCO-MiniLM-L6-v2 Cross-Encoder implementation"""
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the MS-MARCO reranker model"""
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            print(f"Loading MS-MARCO reranker: {self.model_name}")
            self.model = CrossEncoder(self.model_name, activation_fn=torch.nn.Sigmoid())
            print("✓ MS-MARCO reranker loaded successfully")
            
        except ImportError as e:
            print(f"✗ Cannot load MS-MARCO reranker: {e}")
            print("Please install: pip install sentence-transformers torch")
            self.model = None
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Dict]:
        """Rerank documents based on query relevance"""
        if self.model is None:
            return self._mock_rerank(query, documents, top_k)
        
        try:
            # Prepare pairs
            pairs = [(query, doc) for doc in documents]
            
            # Get scores
            scores = self.model.predict(pairs)
            
            # Create results with scores
            results = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                results.append({
                    'document': doc,
                    'score': float(score),
                    'index': i
                })
            
            # Sort by score (higher is better)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            if top_k:
                results = results[:top_k]
            
            return results
            
        except Exception as e:
            print(f"Error in MS-MARCO reranking: {e}")
            return self._mock_rerank(query, documents, top_k)
    
    def _mock_rerank(self, query: str, documents: List[str], top_k: int = None):
        """Mock reranking for when model is not available"""
        import random
        results = []
        for i, doc in enumerate(documents):
            # Simple keyword matching score + random for demo
            keywords = query.lower().split()
            doc_lower = doc.lower()
            keyword_score = sum(1 for kw in keywords if kw in doc_lower)
            mock_score = (keyword_score / len(keywords)) + random.uniform(0, 0.3)
            
            results.append({
                'document': doc,
                'score': mock_score,
                'index': i
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        if top_k:
            results = results[:top_k]
        return results


class CohereReranker:
    """Cohere Rerank API implementation"""
    
    def __init__(self, api_key: str = None, model: str = "rerank-v3.5"):
        self.api_key = api_key or os.getenv('COHERE_API_KEY')
        self.model = model
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Cohere client"""
        try:
            import cohere
            
            if self.api_key:
                self.client = cohere.Client(self.api_key)
                print("✓ Cohere client initialized successfully")
            else:
                print("✗ Cohere API key not provided")
                print("Set COHERE_API_KEY environment variable or pass api_key parameter")
                
        except ImportError:
            print("✗ Cannot load Cohere client")
            print("Please install: pip install cohere")
            self.client = None
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Dict]:
        """Rerank documents using Cohere API"""
        if self.client is None:
            return self._mock_rerank(query, documents, top_k)
        
        try:
            # Call Cohere rerank API
            response = self.client.rerank(
                query=query,
                documents=documents,
                top_n=top_k or len(documents),
                model=self.model
            )
            
            # Convert to our standard format
            results = []
            for result in response.results:
                results.append({
                    'document': documents[result.index],
                    'score': result.relevance_score,
                    'index': result.index
                })
            
            return results
            
        except Exception as e:
            print(f"Error in Cohere reranking: {e}")
            return self._mock_rerank(query, documents, top_k)
    
    def _mock_rerank(self, query: str, documents: List[str], top_k: int = None):
        """Mock reranking for when API is not available"""
        import random
        results = []
        for i, doc in enumerate(documents):
            # Advanced keyword matching simulation
            keywords = query.lower().split()
            doc_lower = doc.lower()
            
            # Calculate relevance score
            keyword_matches = sum(1 for kw in keywords if kw in doc_lower)
            length_penalty = min(1.0, 50.0 / len(doc))  # Prefer shorter, relevant docs
            mock_score = (keyword_matches / len(keywords)) * 0.7 + length_penalty * 0.3 + random.uniform(0, 0.1)
            
            results.append({
                'document': doc,
                'score': mock_score,
                'index': i
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        if top_k:
            results = results[:top_k]
        return results


class RerankerBenchmark:
    """Benchmark suite for comparing rerankers"""
    
    def __init__(self):
        self.results = []
    
    def run_benchmark(self, queries_and_docs: List[Dict], top_k: int = 5):
        """Run benchmark on all rerankers"""
        print("="*80)
        print("RERANKER BENCHMARK SUITE")
        print("="*80)
        
        # Initialize rerankers
        print("\nInitializing Rerankers...")
        print("-" * 40)
        
        bge_reranker = BGEReranker()
        msmarco_reranker = MSMarcoReranker()
        cohere_reranker = CohereReranker()  # Will use mock if no API key
        
        rerankers = {
            "BGE-Reranker-Base": bge_reranker,
            "MS-MARCO-MiniLM-L6-v2": msmarco_reranker,
            "Cohere-Rerank": cohere_reranker
        }
        
        # Run tests
        for test_case in queries_and_docs:
            query = test_case["query"]
            documents = test_case["documents"]
            
            print(f"\n{'='*60}")
            print(f"QUERY: {query}")
            print(f"{'='*60}")
            
            for reranker_name, reranker in rerankers.items():
                print(f"\n{reranker_name}:")
                print("-" * 30)
                
                # Time the reranking
                start_time = time.time()
                results = reranker.rerank(query, documents, top_k=top_k)
                end_time = time.time()
                
                # Display results
                print(f"Processing time: {end_time - start_time:.3f}s")
                print(f"Top {len(results)} results:")
                
                for i, result in enumerate(results):
                    score = result['score']
                    doc = result['document']
                    if len(doc) > 80:
                        doc = doc[:77] + "..."
                    print(f"  {i+1}. [Score: {score:.4f}] {doc}")
                
                # Store results for analysis
                self.results.append({
                    'reranker': reranker_name,
                    'query': query,
                    'processing_time': end_time - start_time,
                    'results': results
                })
        
        self._print_summary()
    
    def _print_summary(self):
        """Print benchmark summary"""
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        
        # Calculate average processing times
        reranker_times = {}
        for result in self.results:
            reranker = result['reranker']
            if reranker not in reranker_times:
                reranker_times[reranker] = []
            reranker_times[reranker].append(result['processing_time'])
        
        print("\nAverage Processing Times:")
        print("-" * 30)
        for reranker, times in reranker_times.items():
            avg_time = sum(times) / len(times)
            print(f"{reranker}: {avg_time:.3f}s")
        
        print("\nRecommendations:")
        print("-" * 30)
        print("• BGE-Reranker-Base: Good balance of performance and speed")
        print("• MS-MARCO-MiniLM-L6-v2: Fast inference, great for real-time applications")
        print("• Cohere-Rerank: Highest accuracy, requires API key, good for production")


def main():
    """Main function to run the reranker tests"""
    print("Reranker Testing Suite")
    print("Testing BGE-Reranker-Base, MS-MARCO-MiniLM-L6-v2, and Cohere Rerank")
    print("\nNote: If models are not installed, mock implementations will be used for demonstration.")
    
    # Create benchmark
    benchmark = RerankerBenchmark()
    
    # Run benchmark
    benchmark.run_benchmark(TEST_QUERIES_AND_DOCS, top_k=5)
    
#     print(f"\n{'='*80}")
#     print("INSTALLATION INSTRUCTIONS")
#     print(f"{'='*80}")
#     print("\nTo run with actual models, install the following:")
#     print("\n1. For BGE and MS-MARCO rerankers:")
#     print("   pip install torch transformers sentence-transformers")
    
#     print("\n2. For Cohere reranker:")
#     print("   pip install cohere")
#     print("   export COHERE_API_KEY='your-api-key-here'")
    
#     print("\n3. Alternative FlagEmbedding installation for BGE:")
#     print("   pip install FlagEmbedding")
    
#     print(f"\n{'='*80}")
#     print("EXAMPLE COHERE IMPLEMENTATION")
#     print(f"{'='*80}")
    
#     print("""
# # Example Cohere reranker usage:
# import cohere

# co = cohere.Client('your-api-key')

# query = "What is machine learning?"
# documents = [
#     "Machine learning is a subset of AI...",
#     "Deep learning uses neural networks...",
#     # ... more documents
# ]

# # Rerank documents
# response = co.rerank(
#     query=query,
#     documents=documents,
#     top_n=5,
#     model="rerank-v3.5"
# )

# # Process results
# for result in response.results:
#     print(f"Score: {result.relevance_score:.4f}")
#     print(f"Document: {documents[result.index]}")
#     print()
# """)


if __name__ == "__main__":
    main()