"""
Retrieval Engine
---------------
Smart document retrieval using hybrid search (BM25 + FAISS) with dynamic chunk sizing
and intelligent information passing based on query complexity.

Key Features:
- Hybrid search combining keyword (BM25) and semantic (FAISS) search
- Dynamic chunk sizing based on query complexity
- Smart metadata selection
- Entity-based reranking
- Intent-based filtering
- Comprehensive citation tracking
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from dataclasses import dataclass
import re
from collections import defaultdict
import torch
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata."""
    text: str
    doc_id: str
    page_num: int
    chunk_id: int
    tables: List[Dict[str, Any]]
    context_before: str
    context_after: str
    metadata: Dict[str, Any]
    
class SearchResult(NamedTuple):
    """Represents a single search result with scores."""
    chunk: DocumentChunk
    semantic_score: float
    keyword_score: float
    combined_score: float
    entity_boost: float

class RetrievalEngine:
    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        cache_dir: str = "cache",
        use_gpu: bool = torch.cuda.is_available()
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize SBERT
        device = 'cuda' if use_gpu else 'cpu'
        self.encoder = SentenceTransformer(model_name, device=device)
        
        # Initialize search components
        self.index = None
        self.bm25 = None
        self.chunks = []
        self.chunk_texts = []
        self.doc_lookup = {}
        
        # Configurable parameters
        self.semantic_weight = 0.7
        self.keyword_weight = 0.3
        self.entity_boost_factor = 1.2
        self.intent_penalty_factor = 0.8

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        return text

    def smart_chunk_size(self, text: str, query_complexity: float) -> int:
        """Determine optimal chunk size based on content and query."""
        base_size = 512
        
        # Adjust for query complexity (0-1 scale)
        size_multiplier = 1 + query_complexity
        
        # Consider document features
        if len(re.findall(r'(?:figure|table|eq\.?)\s+\d+', text, re.I)) > 0:
            size_multiplier *= 1.2  # Larger chunks for content with references
            
        if len(re.findall(r'[A-Z]{2,}(?:\s+\d+)?', text)) > 3:
            size_multiplier *= 1.1  # Larger for technical content
            
        return int(base_size * size_multiplier)

    def extract_context(
        self,
        text: str,
        chunk_start: int,
        chunk_end: int,
        context_size: int = 100
    ) -> Tuple[str, str]:
        """Extract context before and after chunk."""
        context_before = text[max(0, chunk_start - context_size):chunk_start].strip()
        context_after = text[chunk_end:min(len(text), chunk_end + context_size)].strip()
        return context_before, context_after

    def create_chunks(
        self,
        document: Dict[str, Any],
        query_complexity: float
    ) -> List[DocumentChunk]:
        """Create smart document chunks with context."""
        chunks = []
        doc_id = document['metadata']['filename']
        
        for page in document['content']:
            text = page['text']
            page_num = page['page_number']
            
            # Get optimal chunk size
            chunk_size = self.smart_chunk_size(text, query_complexity)
            
            # Create overlapping chunks
            overlap = chunk_size // 4
            start = 0
            chunk_id = 0
            
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                
                if len(chunk_text.strip()) < 50:  # Skip very small chunks
                    break
                    
                # Get context
                context_before, context_after = self.extract_context(
                    text, start, end
                )
                
                # Find relevant tables for this chunk
                chunk_tables = [
                    table for table in page['tables']
                    if self._is_table_relevant(table, start, end)
                ]
                
                # Create chunk
                chunk = DocumentChunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    page_num=page_num,
                    chunk_id=chunk_id,
                    tables=chunk_tables,
                    context_before=context_before,
                    context_after=context_after,
                    metadata=document['metadata']
                )
                
                chunks.append(chunk)
                chunk_id += 1
                start += chunk_size - overlap
                
        return chunks

    def _is_table_relevant(
        self,
        table: Dict[str, Any],
        chunk_start: int,
        chunk_end: int
    ) -> bool:
        """Check if table is relevant to chunk position."""
        table_pos = table['metadata']['bbox']
        table_start = table_pos[1]  # y1 coordinate
        table_end = table_pos[3]    # y2 coordinate
        
        # Check for overlap
        return (chunk_start <= table_end and chunk_end >= table_start)

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        query_complexity: float = 0.5
    ) -> None:
        """Index documents for search."""
        self.chunks = []
        self.chunk_texts = []
        self.doc_lookup = {}
        
        # Create chunks for all documents
        for doc in documents:
            doc_chunks = self.create_chunks(doc, query_complexity)
            self.chunks.extend(doc_chunks)
            
            # Create lookup
            self.doc_lookup[doc['metadata']['filename']] = doc
            
        # Prepare texts for indexing
        self.chunk_texts = [
            self.preprocess_text(chunk.text) for chunk in self.chunks
        ]
        
        # Create FAISS index
        embeddings = self.encoder.encode(
            self.chunk_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.cpu().numpy())
        
        # Create BM25 index
        tokenized_chunks = [text.split() for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        logging.info(f"Indexed {len(self.chunks)} chunks from {len(documents)} documents")

    def estimate_query_complexity(self, query: Dict[str, Any]) -> float:
        """Estimate query complexity for dynamic k selection."""
        complexity_score = 0.0
        
        # Consider query length
        query_text = query.get('query', '')
        words = len(query_text.split())
        complexity_score += min(words / 20, 1.0) * 0.3
        
        # Consider entity requirements
        required_entities = query.get('analysis', {}).get('required_entities', {})
        num_entities = sum(len(entities) for entities in required_entities.values())
        complexity_score += min(num_entities / 5, 1.0) * 0.3
        
        # Consider intent
        intent = query.get('analysis', {}).get('intent', '')
        if intent in ['Technical', 'Regulatory']:
            complexity_score += 0.2
        
        # Consider semantic queries
        semantic_queries = query.get('analysis', {}).get('semantic_queries', [])
        if len(semantic_queries) > 1:
            complexity_score += 0.2
            
        return min(complexity_score, 1.0)

    def dynamic_top_k(self, complexity: float) -> int:
        """Determine number of results based on complexity."""
        base_k = 3
        max_k = 8
        return int(base_k + (max_k - base_k) * complexity)

    def search(
        self,
        query: Dict[str, Any],
        min_score: float = 0.5
    ) -> List[SearchResult]:
        """Perform hybrid search with smart reranking."""
        query_text = query['query']
        query_complexity = self.estimate_query_complexity(query)
        k = self.dynamic_top_k(query_complexity)
        
        # Semantic search
        query_embedding = self.encoder.encode([query_text])[0]
        semantic_scores, indices = self.index.search(
            query_embedding.reshape(1, -1),
            k * 2  # Get more candidates for reranking
        )
        
        # Normalize semantic scores
        semantic_scores = semantic_scores[0]
        semantic_scores = (semantic_scores + 1) / 2  # Convert to 0-1 range
        
        # Keyword search
        tokenized_query = self.preprocess_text(query_text).split()
        keyword_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize keyword scores
        keyword_scores = keyword_scores / np.max(keyword_scores)
        
        results = []
        for idx, semantic_score in zip(indices[0], semantic_scores):
            chunk = self.chunks[idx]
            keyword_score = keyword_scores[idx]
            
            # Calculate base combined score
            combined_score = (
                self.semantic_weight * semantic_score +
                self.keyword_weight * keyword_score
            )
            
            # Apply entity boost
            entity_boost = self._calculate_entity_boost(
                chunk, query['analysis']['required_entities']
            )
            
            # Apply intent filtering
            if not self._check_intent_compatibility(
                chunk, query['analysis']['intent']
            ):
                combined_score *= self.intent_penalty_factor
            
            results.append(SearchResult(
                chunk=chunk,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                combined_score=combined_score * entity_boost,
                entity_boost=entity_boost
            ))
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Filter by minimum score
        results = [r for r in results if r.combined_score >= min_score]
        
        # Limit to top-k
        return results[:k]

    def _calculate_entity_boost(
        self,
        chunk: DocumentChunk,
        required_entities: Dict[str, List[str]]
    ) -> float:
        """Calculate entity match boost factor."""
        boost = 1.0
        chunk_text = chunk.text.lower()
        
        for entity_type, entities in required_entities.items():
            for entity in entities:
                if entity.lower() in chunk_text:
                    boost *= self.entity_boost_factor
                    
        return boost

    def _check_intent_compatibility(
        self,
        chunk: DocumentChunk,
        query_intent: str
    ) -> bool:
        """Check if chunk matches query intent."""
        # Get document intent from metadata
        doc_intent = chunk.metadata.get('intent', '')
        
        # Direct match
        if doc_intent == query_intent:
            return True
            
        # Compatible intents
        compatible_pairs = {
            ('Safety', 'Regulatory'),
            ('Regulatory', 'Technical'),
            ('Equipment', 'Technical')
        }
        
        return (doc_intent, query_intent) in compatible_pairs

    def format_results(
        self,
        results: List[SearchResult],
        include_context: bool = True
    ) -> Dict[str, Any]:
        """Format search results with citations and context."""
        formatted_results = []
        
        for result in results:
            chunk = result.chunk
            
            # Basic citation
            citation = f"[Document: {chunk.doc_id}, Page {chunk.page_num}]"
            
            # Format result
            formatted_result = {
                'text': chunk.text,
                'citation': citation,
                'scores': {
                    'semantic': float(result.semantic_score),
                    'keyword': float(result.keyword_score),
                    'combined': float(result.combined_score),
                    'entity_boost': float(result.entity_boost)
                },
                'metadata': {
                    'doc_id': chunk.doc_id,
                    'page_num': chunk.page_num,
                    'chunk_id': chunk.chunk_id
                }
            }
            
            # Add context if requested
            if include_context:
                formatted_result.update({
                    'context': {
                        'before': chunk.context_before,
                        'after': chunk.context_after
                    }
                })
            
            # Add relevant tables
            if chunk.tables:
                formatted_result['tables'] = chunk.tables
                
            formatted_results.append(formatted_result)
            
        return {
            'results': formatted_results,
            'total_results': len(formatted_results)
        }

def main():
    """Example usage of retrieval engine."""
    try:
        print("\n=== Retrieval Engine Startup ===")
        logging.info("Initializing Retrieval Engine...")
        
        # Check directories
        uploads_dir = Path('uploads')
        processed_dir = Path('processed')
        processed_advanced_dir = Path('processed_advanced')
        
        # Check if initial processing has been done
        if not processed_dir.exists() or not list(processed_dir.glob('*.json')):
            logging.error("\n✗ No processed documents found!")
            logging.error("\nPlease follow these steps:")
            logging.error("1. Create 'uploads' directory and add PDF files")
            logging.error("2. Run: python 1_extract_basic.py")
            logging.error("3. Run: python 2_extract_advanced.py")
            logging.error("4. Then run this script again")
            return
            
        # Check if advanced processing has been done
        if not processed_advanced_dir.exists() or not list(processed_advanced_dir.glob('*.json')):
            logging.error("\n✗ No advanced processed documents found!")
            logging.error("Please run: python 2_extract_advanced.py first")
            return
            
        engine = RetrievalEngine()
        
        logging.info("\nLoading documents from processed_advanced/...")
        documents = []
        for json_file in processed_advanced_dir.glob('*.json'):
            logging.info(f"Reading {json_file.name}")
            with open(json_file, 'r', encoding='utf-8') as f:
                documents.append(json.load(f))
        
        if not documents:
            logging.error("\n✗ No documents found to index!")
            return
            
        logging.info(f"\nIndexing {len(documents)} documents...")
        engine.index_documents(documents)
        
        # Example query
        query = {
            'query': 'What are the safety measures for handling benzene?',
            'analysis': {
                'intent': 'Safety',
                'semantic_queries': [
                    'benzene handling safety protocols',
                    'safety requirements for benzene storage'
                ],
                'keywords': [
                    'benzene', 'safety', 'handling', 'storage', 'protection'
                ],
                'required_entities': {
                    'equipment': ['PPE', 'ventilation'],
                    'standards': ['OSHA', 'API'],
                    'units': ['ppm']
                }
            }
        }
        
        logging.info("\nPerforming search with example query...")
        results = engine.search(query)
        
        if results:
            logging.info(f"\n✓ Found {len(results)} relevant results")
            # Format and save results
            formatted_results = engine.format_results(results)
            
            output_dir = Path('search_results')
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / 'benzene_safety_results.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': query,
                    'results': formatted_results,
                    'metadata': {
                        'total_results': len(results),
                        'query_complexity': engine.estimate_query_complexity(query),
                        'top_k': engine.dynamic_top_k(engine.estimate_query_complexity(query))
                    }
                }, f, ensure_ascii=False, indent=2)
                
            logging.info(f"✓ Results saved to {output_file}")
            
            # Print sample results
            logging.info("\nTop 3 Results:")
            for i, result in enumerate(formatted_results['results'][:3], 1):
                logging.info(f"\n{i}. Document: {result['metadata']['doc_id']}, Page {result['metadata']['page_num']}")
                logging.info(f"Combined Score: {result['scores']['combined']:.3f}")
                logging.info(f"Text: {result['text'][:200]}...")
                
        else:
            logging.error("\n✗ No results found for the query")
            
    except Exception as e:
        logging.error(f"\n✗ Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()