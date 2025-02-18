"""
PDF Extraction and Processing Script
----------------------------------
This script processes PDF documents to extract text, tables, and metadata.

Output Files Generated (in 'processed' directory):
-----------------------------------------------
1. {document_name}.json - Individual document data with full content
2. {document_name}_table_contexts.json - Tables from individual document

Schema Structures:
----------------
1. Individual Document JSON ({document_name}.json):
{
    "metadata": {
        "filename": str,             # Original PDF filename
        "file_path": str,            # Full path to PDF file
        "file_hash": str,            # SHA-256 hash of PDF
        "extraction_date": str,     # ISO format timestamp
        "title": str,               # PDF metadata title
        "author": str,              # PDF metadata author
        "subject": str,             # PDF metadata subject
        "creator": str,             # PDF metadata creator
        "creation_date": str,       # PDF creation date
        "page_count": int,          # Total pages in PDF
        "total_tables": int,        # Total tables found
        "pages_with_tables": int    # Number of pages containing tables
    },
    "content": [
        {
            "page_number": int,   # 1-based page number
            "text": str,          # Extracted text content
            "tables": [           # List of tables on page
                {
                    "metadata": {
                        "table_id": int,                    # Sequential table ID
                        "page_number": int,                 # Page where table appears
                        "bbox": [float, float, float, float],   # Table coordinates
                        "rows": int,                        # Number of rows
                        "columns": int,                     # Number of columns
                        "headers": List[str],               # Column headers
                        "context_headings": List[str],      # Nearby headings
                        "text_before": str,                 # Text preceding table
                        "text_after": str                   # Text following table
                    },
                    "data": List[List[Any]]             # Table content as 2D array
                }
            ]
        }
    ]
}

2. Table Contexts JSON ({document_name}_table_contexts.json):
{
    "file_metadata": {
        "filename": str,           # Original PDF filename
        "extraction_date": str,    # ISO format timestamp
        "total_tables": int        # Number of tables in document
    },
    "tables": [                    # List of all tables (same structure as above)
        {
            "metadata": {...},     # Table metadata
            "data": [...]          # Table content
        }
    ]
}

Usage:
-----
1. Place PDF files in 'uploads' directory
2. Run script: python extract_basic.py
3. Output files will be generated in 'processed' directory
4. Use table.py to visualize extracted tables
"""

# Setup
from datetime import datetime
from pathlib import Path
import json
import PyPDF2
import pdfplumber
import hashlib
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Table metadata
@dataclass
class TableMetadata:
    table_id: int
    page_number: int
    bbox: List[float]
    rows: int
    columns: int
    headers: List[str]
    context_headings: List[str]
    text_before: str
    text_after: str

# Document metadata
@dataclass
class DocumentMetadata:
    filename: str
    file_path: str
    file_hash: str
    extraction_date: str
    title: str
    author: str
    subject: str
    creator: str
    creation_date: str
    page_count: int
    total_tables: int
    pages_with_tables: int

def generate_file_hash(file_path: Path) -> str:
    """Generate SHA-256 hash of file for tracking changes."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def extract_pdf_metadata(pdf_reader: PyPDF2.PdfReader) -> Dict[str, Any]:
    """Extract basic PDF metadata."""
    metadata = pdf_reader.metadata if pdf_reader.metadata else {}
    return {
        'title': metadata.get('/Title', ''),
        'author': metadata.get('/Author', ''),
        'subject': metadata.get('/Subject', ''),
        'creator': metadata.get('/Creator', ''),
        'creation_date': metadata.get('/CreationDate', ''),
        'page_count': len(pdf_reader.pages)
    }

def get_table_context(page: Any, table_bbox: Tuple[float, float, float, float], 
                     text_blocks: List[Dict]) -> Dict[str, Any]:
    """Extract context around a table including headings and surrounding text."""
    table_top, table_bottom = table_bbox[1], table_bbox[3]
    context = {
        'potential_headings': [],
        'text_before': [],
        'text_after': []
    }
    
    for block in text_blocks:
        block_bbox = block.get('bbox')
        if not block_bbox:
            continue
            
        block_bottom = block_bbox[3]
        block_top = block_bbox[1]
        text = block.get('text', '').strip()
        
        if not text:
            continue
        
        # Text blocks just above the table (potential headings)
        if block_bottom < table_top and block_bottom >= table_top - 50:
            context['potential_headings'].append({
                'text': text,
                'distance': table_top - block_bottom
            })
        # Text before the table
        elif block_bottom < table_top:
            context['text_before'].append(text)
        # Text after the table
        elif block_top > table_bottom:
            context['text_after'].append(text)
    
    # Sort and limit headings by distance
    context['potential_headings'].sort(key=lambda x: x['distance'])
    headings = [h['text'] for h in context['potential_headings'][:2]]
    
    return {
        'headings': headings,
        'text_before': ' '.join(context['text_before'][-2:]),
        'text_after': ' '.join(context['text_after'][:2])
    }

def process_table(table: List[List[Any]], table_bbox: Tuple[float, float, float, float], 
                 page: Any, page_num: int, table_idx: int) -> Dict[str, Any]:
    """Process a single table and its context."""
    # Convert table to DataFrame
    df = pd.DataFrame(table)
    df = df.replace('', None).dropna(how='all', axis=1).dropna(how='all', axis=0)
    df.columns = [f'Column_{i+1}' for i in range(len(df.columns))]
    
    # Get text blocks for context
    text_blocks = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
    context = get_table_context(page, table_bbox, text_blocks)
    
    # Create table metadata
    table_metadata = TableMetadata(
        table_id=table_idx + 1,
        page_number=page_num,
        bbox=list(table_bbox),
        rows=len(df),
        columns=len(df.columns),
        headers=df.columns.tolist(),
        context_headings=context['headings'],
        text_before=context['text_before'],
        text_after=context['text_after']
    )
    
    return {
        'metadata': asdict(table_metadata),
        'data': df.values.tolist()
    }

def extract_page_content(page: Any, page_num: int) -> Dict[str, Any]:
    """Extract content from a single page."""
    tables = []
    page_tables = page.extract_tables()
    
    if page_tables:
        table_positions = page.find_tables()
        for idx, (table, position) in enumerate(zip(page_tables, table_positions)):
            table_data = process_table(table, position.bbox, page, page_num, idx)
            tables.append(table_data)
    
    return {
        'page_number': page_num,
        'text': page.extract_text().strip(),
        'tables': tables
    }

def extract_document_content(pdf_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Extract content and metadata from a PDF file."""
    try:
        # Extract basic metadata
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            basic_metadata = extract_pdf_metadata(pdf_reader)
        
        # Extract detailed content
        with pdfplumber.open(pdf_path) as pdf:
            content = []
            all_tables = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_content = extract_page_content(page, page_num)
                content.append(page_content)
                all_tables.extend(page_content['tables'])
            
            # Create document metadata
            doc_metadata = DocumentMetadata(
                filename=pdf_path.name,
                file_path=str(pdf_path),
                file_hash=generate_file_hash(pdf_path),
                extraction_date=datetime.now().isoformat(),
                title=basic_metadata['title'],
                author=basic_metadata['author'],
                subject=basic_metadata['subject'],
                creator=basic_metadata['creator'],
                creation_date=basic_metadata['creation_date'],
                page_count=basic_metadata['page_count'],
                total_tables=len(all_tables),
                pages_with_tables=sum(1 for page in content if page['tables'])
            )
            
            # Create main document structure
            document = {
                'metadata': asdict(doc_metadata),
                'content': content
            }
            
            # Create table-specific structure
            tables_document = {
                'file_metadata': {
                    'filename': pdf_path.name,
                    'extraction_date': datetime.now().isoformat(),
                    'total_tables': len(all_tables)
                },
                'tables': all_tables
            }
            
            return document, tables_document
            
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {str(e)}")
        return None, None

def process_directory(input_dir: str = 'uploads', output_dir: str = 'processed') -> None:
    """Process all PDFs in the input directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    pdf_files = list(input_path.glob('*.pdf'))
    all_documents = []
    all_tables = []
    
    for pdf_path in pdf_files:
        logging.info(f"Processing {pdf_path.name}")
        
        document, tables = extract_document_content(pdf_path)
        if document and tables:
            # Save individual document results
            output_base = output_path / pdf_path.stem
            with open(f"{output_base}.json", 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            with open(f"{output_base}_table_contexts.json", 'w', encoding='utf-8') as f:
                json.dump(tables, f, ensure_ascii=False, indent=2)
            
            all_documents.append(document)
            all_tables.append(tables)
    
    # Save collection summaries
    collection_summary = {
        'collection_metadata': {
            'total_documents': len(all_documents),
            'processing_date': datetime.now().isoformat(),
            'input_directory': str(input_path),
            'output_directory': str(output_path),
            'total_tables': sum(doc['metadata']['total_tables'] for doc in all_documents)
        },
        'documents': all_documents
    }
    
    tables_summary = {
        'collection_metadata': {
            'total_documents': len(all_documents),
            'processing_date': datetime.now().isoformat(),
            'total_tables': sum(tables['file_metadata']['total_tables'] for tables in all_tables)
        },
        'documents': all_tables
    }
    
    with open(output_path / 'collection_summary.json', 'w', encoding='utf-8') as f:
        json.dump(collection_summary, f, ensure_ascii=False, indent=2)
    with open(output_path / 'all_table_contexts.json', 'w', encoding='utf-8') as f:
        json.dump(tables_summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_directory()