"""
Advanced Metadata Extraction Script
---------------------------------
This script processes JSON files from basic extraction to add LLM-generated metadata
using Gemini's chat functionality.

Input: JSON files in 'processed' directory (from extract_basic.py)
Output: Enhanced JSON files in 'processed_advanced' directory
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, TypedDict
from datetime import datetime
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Type definitions
class Entities(TypedDict):
    equipment: List[str]
    standards: List[str]
    units: List[str]

class AdvancedMetadata(TypedDict):
    intent: str
    semantic_queries: List[str]
    keywords: List[str]
    entities: Entities
    key_sections: List[str]

# Constants
SYSTEM_PROMPT = """You are a specialized oil/gas document analyst tasked with extracting structured metadata from technical documents. 

IMPORTANT: You must respond with ONLY a valid JSON object and no other text. The JSON object should contain these components:

1. Intent Classification (exactly one): "Safety", "Equipment", "Regulatory", "Technical", or "Other"
2. Semantic Search Queries (exactly 3)
3. Keywords/Phrases (exactly 10)
4. Entities:
   - equipment: List equipment names/types
   - standards: List industry standards
   - units: List measurement units
5. Key Sections: List main document sections

Return ONLY valid JSON matching this schema:
{
  "intent": "string",
  "semantic_queries": ["string", "string", "string"],
  "keywords": ["string", "string", "string", "string", "string", "string", "string", "string", "string", "string"],
  "entities": {
    "equipment": ["string"],
    "standards": ["string"],
    "units": ["string"]
  },
  "key_sections": ["string"]
}"""

def setup_gemini(api_key: str) -> genai.GenerativeModel:
    """Initialize and return Gemini model."""
    genai.configure(api_key=api_key)
    
    # Configure the model with recommended settings
    generation_config = {
        "temperature": 0.7,  # Slightly lower than default for more focused outputs
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config
    )

def validate_metadata(metadata: Dict[str, Any]) -> None:
    """Validate metadata structure and content."""
    required_fields = {
        'intent': str,
        'semantic_queries': list,
        'keywords': list,
        'entities': dict,
        'key_sections': list
    }
    
    # Check required fields and types
    for field, expected_type in required_fields.items():
        if field not in metadata:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(metadata[field], expected_type):
            raise TypeError(f"Invalid type for {field}")
    
    # Validate specific constraints
    if metadata['intent'] not in {'Safety', 'Equipment', 'Regulatory', 'Technical', 'Other'}:
        raise ValueError("Invalid intent classification")
    
    if len(metadata['semantic_queries']) != 3:
        raise ValueError("Must have exactly 3 semantic queries")
        
    if len(metadata['keywords']) != 10:
        raise ValueError("Must have exactly 10 keywords")
        
    required_entity_types = {'equipment', 'standards', 'units'}
    if not all(et in metadata['entities'] for et in required_entity_types):
        raise ValueError("Missing required entity types")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_metadata(model: genai.GenerativeModel, text: str) -> Dict[str, Any]:
    """Extract metadata from document text using Gemini chat."""
    try:
        # Start a new chat session
        chat = model.start_chat(history=[])
        
        # Send system prompt first
        chat.send_message(SYSTEM_PROMPT)
        
        # Send document text and get response
        response = chat.send_message(f"Document Text:\n{text}")
        
        # Print raw response for debugging
        logging.info(f"Raw response:\n{response.text}")
        
        # Try to find JSON in the response
        response_text = response.text.strip()
        try:
            # First try direct JSON parsing
            metadata = json.loads(response_text)
        except json.JSONDecodeError:
            # If that fails, try to find JSON-like structure
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx + 1]
                try:
                    metadata = json.loads(json_str)
                except json.JSONDecodeError:
                    logging.error(f"Could not extract valid JSON from response")
                    raise
            else:
                logging.error("No JSON structure found in response")
                raise ValueError("Response does not contain JSON structure")
        
        # Validate response structure
        validate_metadata(metadata)
        
        return metadata
        
    except Exception as e:
        logging.error(f"Error in metadata extraction: {str(e)}")
        raise

def truncate_text(text: str, max_length: int = 30000) -> str:
    """Truncate text to max_length while keeping whole sentences."""
    if len(text) <= max_length:
        return text
        
    # Find the last sentence boundary before max_length
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    if last_period != -1:
        return truncated[:last_period + 1]
    return truncated

def process_document(
    file_path: Path,
    model: genai.GenerativeModel,
    output_dir: Path
) -> None:
    """Process a single document."""
    try:
        # Load original document
        with open(file_path, 'r', encoding='utf-8') as f:
            document = json.load(f)
        
        # Extract text content from all pages
        full_text = ' '.join(
            page['text'] for page in document['content']
            if page.get('text')
        )
        
        # Truncate text if too long
        full_text = truncate_text(full_text)
        
        # Extract advanced metadata
        advanced_metadata = extract_metadata(model, full_text)
        
        # Add advanced metadata to document
        document['advanced_metadata'] = advanced_metadata
        
        # Save enhanced document
        output_path = output_dir / file_path.name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
            
        logging.info(f"Successfully processed {file_path.name}")
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        raise

def process_directory(
    input_dir: str = 'processed',
    output_dir: str = 'processed_advanced',
    api_key: str = None
) -> None:
    """Process all JSON files in input directory."""
    if not api_key:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

    # Setup directories
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Setup Gemini
    model = setup_gemini(api_key)
    
    # Process files
    json_files = list(input_path.glob('*.json'))
    for file_path in json_files:
        logging.info(f"Processing {file_path.name}")
        process_document(file_path, model, output_path)

def main():
    """Main execution function."""
    try:
        process_directory()
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()