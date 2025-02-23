"""
LLM Layer
---------
Assembles context from retrieved chunks and generates final answers using Gemini.

Key Features:
- Context assembly with proper citations
- LLM query generation with system prompts
- Response validation
- Answer persistence
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
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

class LLMLayer:
    def __init__(self, api_key: str = None):
        """Initialize LLM layer with API key."""
        if not api_key:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
                
        # Initialize Gemini
        genai.configure(api_key=api_key)
        
        # Configure the model
        generation_config = {
            "temperature": 0.3,  # Lower temperature for more focused responses
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config
        )
        
        # Create output directory
        self.output_dir = Path('answers')
        self.output_dir.mkdir(exist_ok=True)

    def assemble_context(self, results: Dict[str, Any]) -> str:
        context_parts = []
        
        # Group results by document
        doc_groups = {}
        for result in results['results']:
            doc_id = result['metadata']['doc_id']
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(result)
        
        # Assemble context with clear document separation
        for doc_id, doc_results in doc_groups.items():
            context_parts.append(f"\n=== From Document: {doc_id} ===\n")
            
            for result in doc_results:
                # Format citation
                citation = result['citation']
                
                # Get text and clean it
                text = result['text'].strip()
                text = re.sub(r'\s+', ' ', text)
                
                # Combine with clear source marking
                context_parts.append(f"{citation}\n{text}\n")
                
                # Add table information if present
                if 'tables' in result and result['tables']:
                    for table in result['tables']:
                        table_text = self._format_table(table)
                        if table_text:
                            context_parts.append(f"{citation} [Table]\n{table_text}\n")
        
        return "\n".join(context_parts)

    def _format_table(self, table: Dict[str, Any]) -> Optional[str]:
        """Format table data into readable text."""
        try:
            metadata = table['metadata']
            data = table['data']
            
            if not data or not data[0]:  # Check if table is empty
                return None
                
            # Format headers
            headers = data[0]
            header_row = " | ".join(str(h) for h in headers)
            
            # Format data rows
            rows = []
            for row in data[1:]:  # Skip header row
                row_text = " | ".join(str(cell) for cell in row)
                rows.append(row_text)
                
            # Combine all parts
            table_text = f"Headers: {header_row}\n"
            table_text += "Data:\n" + "\n".join(rows)
            
            return table_text
            
        except Exception as e:
            logging.warning(f"Error formatting table: {str(e)}")
            return None

    def create_system_prompt(self, context: str, query: str) -> str:
        return f'''You are a technical assistant for engineers in oil/gas industry. Answer the query using ONLY the context below. Important guidelines:

1. Use information ONLY from the provided context
2. Always cite sources using [Document: Name, Page X]
3. If multiple documents are provided, prefer information from documents specifically mentioned in the query
4. If the query asks about a specific substance or topic, prioritize information from documents about that specific topic
5. If unsure or if information is not in the context, clearly state so

Context:
{context}

Query: {query}
Answer:'''

    def validate_response(self, response: str, context: str) -> bool:
        """Validate LLM response for citations and numerical accuracy."""
        # Check for citations
        citations = re.findall(r'\[Document:.*?\]', response)
        if not citations:
            logging.warning("Response missing citations")
            return False
            
        # Validate that citations exist in context
        for citation in citations:
            if citation not in context:
                logging.warning(f"Invalid citation found: {citation}")
                return False
                
        # Check for numerical consistency
        numbers_in_response = re.findall(r'\d+(?:\.\d+)?', response)
        for number in numbers_in_response:
            # Check if number appears in context
            if number not in context:
                logging.warning(f"Number {number} not found in context")
                return False
                
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_answer(self, query: str, results: Dict[str, Any]) -> str:
        """Generate answer using Gemini."""
        try:
            # Assemble context
            context = self.assemble_context(results)
            
            # Create prompt
            prompt = self.create_system_prompt(context, query)
            
            # Get response from Gemini
            chat = self.model.start_chat(history=[])
            response = chat.send_message(prompt)
            
            answer = response.text.strip()
            
            # Validate response
            if not self.validate_response(answer, context):
                raise ValueError("Response validation failed")
                
            return answer
            
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            raise

    def save_answer(self, query: str, answer: str, filename: str = None) -> None:
        """Save query and answer to file."""
        try:
            if not filename:
                # Generate filename from query
                query_slug = re.sub(r'[^\w\s-]', '', query.lower())
                query_slug = re.sub(r'[-\s]+', '_', query_slug)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{query_slug}_{timestamp}.txt"
            
            output_path = self.output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n\n")
                f.write(f"Answer:\n{answer}\n")
                
            logging.info(f"Answer saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Error saving answer: {str(e)}")
            raise

def main():
    """Example usage of LLM layer."""
    try:
        # Initialize LLM layer
        llm = LLMLayer()
        
        # Load example query from queries directory
        queries_dir = Path('queries')
        if not queries_dir.exists():
            raise FileNotFoundError("queries directory not found")
            
        query_file = queries_dir / 'benzene_query.json'
        if not query_file.exists():
            raise FileNotFoundError(f"Query file not found: {query_file}")
            
        # Load results from search_results directory
        search_results_dir = Path('search_results')
        if not search_results_dir.exists():
            raise FileNotFoundError("search_results directory not found")
            
        results_file = search_results_dir / 'benzene_safety_results.json'
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
            
        with open(query_file, 'r') as f:
            query_data = json.load(f)
            
        with open(results_file, 'r') as f:
            results_data = json.load(f)
            
        # Generate answer
        answer = llm.generate_answer(
            query_data['query'],
            results_data['results']
        )
        
        # Print answer
        print("\nQuery:", query_data['query'])
        print("\nAnswer:", answer)
        
        # Save answer
        llm.save_answer(query_data['query'], answer)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()