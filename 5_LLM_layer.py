"""
LLM Answer Generation Layer
-------------------------
Generates final answers with citations using retrieved context chunks.

Input: Retrieval package containing:
- Query analysis
- Relevant document chunks with citations
- Token count

Output: Structured response with:
- Answer text
- Source citations
- Confidence score
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, TypedDict
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GeneratedAnswer(TypedDict):
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float

# Constants
SYSTEM_PROMPT = """You are a technical documentation assistant for oil/gas industry documents.
Your task is to answer questions using the provided context chunks.

IMPORTANT GUIDELINES:
1. Use ONLY information from the provided chunks
2. Always include relevant citations
3. If the context is insufficient, say so
4. Be clear and concise
5. Format citations as: [Document: {name}, Page: {number}]

Maintain technical accuracy while being accessible."""

def setup_gemini(
    api_key: str,
    temperature: float = 0.5,
    top_p: float = 0.95,
    top_k: int = 40
) -> genai.GenerativeModel:
    """Initialize and return Gemini model with specified parameters."""
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config
    )

def prepare_context(retrieval_package: Dict[str, Any]) -> str:
    """Prepare context string from retrieval package."""
    chunks = retrieval_package['chunks']
    query = retrieval_package['query_analysis']
    
    context_parts = ["CONTEXT CHUNKS:"]
    
    for i, chunk in enumerate(chunks, 1):
        citation = chunk['citation']
        context_parts.append(
            f"\nChunk {i}:\n"
            f"Source: {citation['document']}, Page: {citation['page']}\n"
            f"Content: {chunk['text']}\n"
            f"---"
        )
    
    return '\n'.join(context_parts)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_answer(
    model: genai.GenerativeModel,
    query: str,
    context: str
) -> GeneratedAnswer:
    """Generate answer using Gemini chat."""
    try:
        # Start chat session
        chat = model.start_chat(history=[])
        
        # Send system prompt
        chat.send_message(SYSTEM_PROMPT)
        
        # Send context and query
        prompt = f"""
Context Information:
{context}

User Query: {query}

Please provide a clear and concise answer using only the information from the context chunks.
Include relevant citations in the format [Document: name, Page: number].
"""
        
        response = chat.send_message(prompt)
        
        # Process response to extract citations
        answer_text = response.text
        citations = []
        
        # Extract citations using regex pattern
        import re
        citation_pattern = r'\[Document: (.*?), Page: (\d+)\]'
        for match in re.finditer(citation_pattern, answer_text):
            citations.append({
                'document': match.group(1),
                'page': int(match.group(2))
            })
        
        # Calculate simple confidence score based on citation count
        confidence = min(len(citations) / 3, 1.0)  # Scale 0-1, max at 3+ citations
        
        return {
            'answer': answer_text,
            'citations': citations,
            'confidence': confidence
        }
        
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        raise

def process_query(
    retrieval_package: Dict[str, Any],
    api_key: str = None,
    temperature: float = 0.3
) -> GeneratedAnswer:
    """Process retrieval package and generate answer."""
    if not api_key:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

    # Setup model
    model = setup_gemini(api_key, temperature=temperature)
    
    # Prepare context
    context = prepare_context(retrieval_package)
    
    # Generate answer
    return generate_answer(
        model,
        retrieval_package['query_analysis']['semantic_queries'][0],
        context
    )

def main():
    """Example usage of LLM layer."""
    try:
        # Load example retrieval package
        retrieval_dir = Path('retrieval_results')
        retrieval_file = retrieval_dir / 'benzene_retrieval.json'
        
        with open(retrieval_file, 'r', encoding='utf-8') as f:
            retrieval_package = json.load(f)
            
        # Generate answer
        answer_package = process_query(retrieval_package)
        
        # Save results
        output_dir = Path('answers')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / 'benzene_answer.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(answer_package, f, ensure_ascii=False, indent=2)
            
        logging.info(f"Generated answer with {len(answer_package['citations'])} citations")
        logging.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()