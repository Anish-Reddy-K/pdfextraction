"""
Retrieval Engine
---------------
Prepares context packages for LLM by selecting relevant document chunks
based on semantic and keyword matching.

Configuration:
- CHUNK_SIZE: Number of characters per chunk
- NUM_CHUNKS: Maximum number of chunks to return
- OVERLAP_SIZE: Number of characters to overlap between chunks
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, TypedDict
from dataclasses import dataclass
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration - Can be modified as needed
CHUNK_SIZE = 3000  # Characters per chunk
NUM_CHUNKS = 10    # Maximum chunks to return
OVERLAP_SIZE = 200 # Characters of overlap between chunks

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with metadata."""
    text: str
    doc_name: str
    page_number: int
    chunk_number: int
    score: float = 0.0

class RetrievalPackage(TypedDict):
    """Structure for the final context package."""
    chunks: List[Dict[str, Any]]
    query_analysis: Dict[str, Any]
    total_tokens: int

def create_chunks(text: str, doc_name: str, page_number: int) -> List[DocumentChunk]:
    """Split text into overlapping chunks."""
    chunks = []
    
    # Handle empty or None text
    if not text:
        return chunks
        
    # Split into chunks with overlap
    start = 0
    chunk_number = 1
    
    while start < len(text):
        # Calculate chunk end with overlap
        end = start + CHUNK_SIZE
        
        # Adjust end to nearest sentence boundary if possible
        if end < len(text):
            sentence_end = text.find('.', end)
            if sentence_end != -1 and sentence_end - end < 100:
                end = sentence_end + 1
        
        # Create chunk
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(DocumentChunk(
                text=chunk_text,
                doc_name=doc_name,
                page_number=page_number,
                chunk_number=chunk_number
            ))
            
        # Move start position for next chunk
        start = end - OVERLAP_SIZE
        chunk_number += 1
        
    return chunks

def score_chunk(chunk: DocumentChunk, keywords: List[str], semantic_queries: List[str]) -> float:
    """Score chunk relevance using simple keyword and semantic matching."""
    score = 0.0
    text = chunk.text.lower()
    
    # Keyword matching
    for keyword in keywords:
        keyword = keyword.lower()
        count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
        score += count * 0.5  # Weight for keyword matches
    
    # Simple semantic matching using query phrases
    for query in semantic_queries:
        query = query.lower()
        words = query.split()
        for word in words:
            count = len(re.findall(r'\b' + re.escape(word) + r'\b', text))
            score += count * 0.3  # Weight for semantic matches
            
    return score

def retrieve_chunks(
    processed_dir: Path,
    query_analysis: Dict[str, Any],
    chunk_size: int = CHUNK_SIZE,
    num_chunks: int = NUM_CHUNKS
) -> RetrievalPackage:
    """Retrieve and score relevant chunks based on query analysis."""
    all_chunks = []
    
    # Process each document in the processed_advanced directory
    for json_file in processed_dir.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                document = json.load(f)
            
            # Process each page
            for page in document['content']:
                chunks = create_chunks(
                    page['text'],
                    document['metadata']['filename'],
                    page['page_number']
                )
                
                # Score chunks
                for chunk in chunks:
                    chunk.score = score_chunk(
                        chunk,
                        query_analysis['keywords'],
                        query_analysis['semantic_queries']
                    )
                    
                all_chunks.extend(chunks)
                
        except Exception as e:
            logging.error(f"Error processing {json_file}: {str(e)}")
            continue
    
    # Sort chunks by score and take top N
    sorted_chunks = sorted(all_chunks, key=lambda x: x.score, reverse=True)
    top_chunks = sorted_chunks[:num_chunks]
    
    # Prepare retrieval package
    chunk_dicts = [
        {
            'text': chunk.text,
            'citation': {
                'document': chunk.doc_name,
                'page': chunk.page_number,
                'chunk': chunk.chunk_number
            },
            'score': chunk.score
        }
        for chunk in top_chunks
    ]
    
    # Estimate total tokens (rough approximation)
    total_tokens = sum(len(chunk.text.split()) for chunk in top_chunks)
    
    return {
        'chunks': chunk_dicts,
        'query_analysis': query_analysis,
        'total_tokens': total_tokens
    }

def main():
    """Example usage of retrieval engine."""
    try:
        # Load example query analysis
        queries_dir = Path('queries')
        query_file = queries_dir / 'benzene_query.json'
        
        with open(query_file, 'r', encoding='utf-8') as f:
            query_data = json.load(f)
            
        # Process query
        processed_dir = Path('processed_advanced')
        retrieval_package = retrieve_chunks(processed_dir, query_data['analysis'])
        
        # Save results
        output_dir = Path('retrieval_results')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / 'benzene_retrieval.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(retrieval_package, f, ensure_ascii=False, indent=2)
            
        logging.info(f"Retrieved {len(retrieval_package['chunks'])} chunks")
        logging.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()