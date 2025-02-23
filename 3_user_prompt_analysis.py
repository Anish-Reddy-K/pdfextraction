"""
User Prompt Analysis Script
-------------------------
This script analyzes user queries to determine intent and extract key components
for document retrieval.

Input: User query text
Output: Structured query information for the retrieval engine

Query Analysis Schema:
-------------------
{
    "intent": str,          # Query intent classification
    "semantic_queries": [   # List of 2 rephrased search queries
        str,
        str
    ],
    "keywords": [          # List of 5 key terms
        str,
        str,
        str,
        str,
        str
    ],
    "required_entities": {  # Required entities by type
        "equipment": [str],
        "standards": [str],
        "units": [str]
    }
}
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, TypedDict
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from dotenv import load_dotenv
import warnings

# Load environment variables
load_dotenv()

# Suppress gRPC cleanup warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.generativeai')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Type definitions
class RequiredEntities(TypedDict):
    equipment: List[str]
    standards: List[str]
    units: List[str]

class QueryAnalysis(TypedDict):
    intent: str
    semantic_queries: List[str]
    keywords: List[str]
    required_entities: RequiredEntities

# Constants
SYSTEM_PROMPT = """You are a specialized query analyzer for oil/gas technical documentation systems.

IMPORTANT: You must respond with ONLY a valid JSON object and no other text. Analyze the user's query and return a JSON object with:

1. Intent Classification (exactly one): "Safety", "Equipment", "Regulatory", "Technical", or "Other"
2. Semantic Search Queries (exactly 2): Rephrase the query for better retrieval
3. Keywords (exactly 5): Key terms for search
4. Required Entities: Extract mentions of:
   - equipment: List equipment names/types
   - standards: List industry standards (e.g., API codes)
   - units: List measurement units

The JSON must match this schema exactly:
{
  "intent": "string",
  "semantic_queries": ["string", "string"],
  "keywords": ["string", "string", "string", "string", "string"],
  "required_entities": {
    "equipment": ["string"],
    "standards": ["string"],
    "units": ["string"]
  }
}"""

def setup_gemini(api_key: str) -> genai.GenerativeModel:
    """Initialize and return Gemini model."""
    genai.configure(api_key=api_key)
    
    # Configure the model with recommended settings
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config
    )

def validate_query_analysis(analysis: Dict[str, Any]) -> None:
    """Validate query analysis structure and content."""
    # Check base structure
    required_fields = {
        'intent': str,
        'semantic_queries': list,
        'keywords': list,
        'required_entities': dict
    }
    
    for field, expected_type in required_fields.items():
        if field not in analysis:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(analysis[field], expected_type):
            raise TypeError(f"Invalid type for {field}")
    
    # Validate specific constraints
    if analysis['intent'] not in {'Safety', 'Equipment', 'Regulatory', 'Technical', 'Other'}:
        raise ValueError("Invalid intent classification")
    
    if len(analysis['semantic_queries']) != 2:
        raise ValueError("Must have exactly 2 semantic queries")
        
    if len(analysis['keywords']) != 5:
        raise ValueError("Must have exactly 5 keywords")
        
    required_entity_types = {'equipment', 'standards', 'units'}
    if not all(et in analysis['required_entities'] for et in required_entity_types):
        raise ValueError("Missing required entity types")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_query(model: genai.GenerativeModel, query: str) -> QueryAnalysis:
    """Analyze user query using Gemini chat."""
    try:
        # Start a new chat session
        chat = model.start_chat(history=[])
        
        # Send system prompt first
        chat.send_message(SYSTEM_PROMPT)
        
        # Send user query and get response
        response = chat.send_message(f"User Query:\n{query}")
        
        # Print raw response for debugging
        logging.debug(f"Raw response:\n{response.text}")
        
        # Try to find JSON in the response
        response_text = response.text.strip()
        try:
            # First try direct JSON parsing
            analysis = json.loads(response_text)
        except json.JSONDecodeError:
            # If that fails, try to find JSON-like structure
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx + 1]
                try:
                    analysis = json.loads(json_str)
                except json.JSONDecodeError:
                    logging.error(f"Could not extract valid JSON from response")
                    raise
            else:
                logging.error("No JSON structure found in response")
                raise ValueError("Response does not contain JSON structure")
        
        # Validate analysis structure
        validate_query_analysis(analysis)
        
        return analysis
        
    except Exception as e:
        logging.error(f"Error in query analysis: {str(e)}")
        raise

def process_user_query(query: str, api_key: str = None) -> QueryAnalysis:
    """Process a user query and return structured analysis."""
    if not api_key:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

    # Setup Gemini
    model = setup_gemini(api_key)
    
    # Analyze query
    return analyze_query(model, query)

def main():
    """Main execution function with example usage."""
    try:
        # Test query for benzene handling
        query = "What is the transport information for benzene?"
        
        # Create queries directory if it doesn't exist
        queries_dir = Path('queries')
        queries_dir.mkdir(exist_ok=True)
        
        # Process query and save results
        logging.info(f"\nAnalyzing query: {query}")
        analysis = process_user_query(query)
        
        # Save to JSON file
        output_file = queries_dir / 'benzene_query.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'query': query,
                'analysis': analysis
            }, f, ensure_ascii=False, indent=2)
            
        logging.info(f"Analysis saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()