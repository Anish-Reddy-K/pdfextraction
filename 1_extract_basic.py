"""
PDF Extraction and Processing Script
----------------------------------
This script processes PDF documents to extract text, tables, and metadata.

Output Files Generated (in 'processed' directory):
-----------------------------------------------
{document_name}.json - Complete document data including content, tables, and metadata

Schema Structure:
----------------
Document JSON ({document_name}.json):
{
    "metadata": {
        "filename": str,          # Original PDF filename
        "file_path": str,         # Full path to PDF file
        "file_hash": str,         # SHA-256 hash of PDF
        "extraction_date": str,    # ISO format timestamp
        "title": str,             # PDF metadata title
        "author": str,            # PDF metadata author
        "subject": str,           # PDF metadata subject
        "creator": str,           # PDF metadata creator
        "creation_date": str,     # PDF creation date
        "page_count": int,        # Total pages in PDF
        "total_tables": int,      # Total tables found
        "pages_with_tables": int  # Number of pages containing tables
    },
    "content": [
        {
            "page_number": int,   # 1-based page number
            "text": str,          # Extracted text content
            "tables": [           # List of tables on page
                {
                    "metadata": {
                        "table_id": int,               # Sequential table ID
                        "page_number": int,            # Page where table appears
                        "bbox": [float, float, float, float],  # Table coordinates
                        "rows": int,                   # Number of rows
                        "columns": int,                # Number of columns
                        "headers": List[str],          # Column headers
                        "context_headings": List[str], # Nearby headings
                        "text_before": str,            # Text preceding table
                        "text_after": str              # Text following table
                    },
                    "data": List[List[Any]]           # Table content as 2D array
                }
            ]
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
        
        if block_bottom < table_top and block_bottom >= table_top - 50:
            context['potential_headings'].append({
                'text': text,
                'distance': table_top - block_bottom
            })
        elif block_bottom < table_top:
            context['text_before'].append(text)
        elif block_top > table_bottom:
            context['text_after'].append(text)
    
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
    df = pd.DataFrame(table)
    df = df.replace('', None).dropna(how='all', axis=1).dropna(how='all', axis=0)
    df.columns = [f'Column_{i+1}' for i in range(len(df.columns))]
    
    text_blocks = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
    context = get_table_context(page, table_bbox, text_blocks)
    
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

def extract_document_content(pdf_path: Path) -> Optional[Dict[str, Any]]:
    """Extract content and metadata from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            basic_metadata = extract_pdf_metadata(pdf_reader)
        
        with pdfplumber.open(pdf_path) as pdf:
            content = []
            all_tables = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_content = extract_page_content(page, page_num)
                content.append(page_content)
                all_tables.extend(page_content['tables'])
            
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
            
            return {
                'metadata': asdict(doc_metadata),
                'content': content
            }
            
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {str(e)}")
        return None

def process_directory(input_dir: str = 'uploads', output_dir: str = 'processed') -> None:
    """Process all PDFs in the input directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    pdf_files = list(input_path.glob('*.pdf'))
    
    for pdf_path in pdf_files:
        logging.info(f"Processing {pdf_path.name}")
        
        document = extract_document_content(pdf_path)
        if document:
            output_file = output_path / f"{pdf_path.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            logging.info(f"Successfully processed {pdf_path.name}")
        else:
            logging.error(f"Failed to process {pdf_path.name}")

if __name__ == "__main__":
    process_directory()