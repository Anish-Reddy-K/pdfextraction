import os
from datetime import datetime
from pathlib import Path
import json
import PyPDF2
import pdfplumber
import hashlib
import pandas as pd
from tqdm import tqdm

class PDFParser:
    def __init__(self, input_dir='uploads', output_dir='processed'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_file_hash(self, file_path):
        """Generate SHA-256 hash of file for tracking changes."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def extract_pdf_metadata(self, pdf_reader):
        """Extract PDF metadata from the document."""
        metadata = pdf_reader.metadata if pdf_reader.metadata else {}
        return {
            'title': metadata.get('/Title', ''),
            'author': metadata.get('/Author', ''),
            'subject': metadata.get('/Subject', ''),
            'creator': metadata.get('/Creator', ''),
            'creation_date': metadata.get('/CreationDate', ''),
            'page_count': len(pdf_reader.pages)
        }

    def extract_tables_from_page(self, page):
        """Extract tables from a single page using pdfplumber."""
        tables = page.extract_tables()
        processed_tables = []
        
        if tables:
            for table_idx, table in enumerate(tables, 1):
                # Convert table to pandas DataFrame for better processing
                df = pd.DataFrame(table)
                
                # Try to use first row as header if it looks like a header
                if df.shape[0] > 1:
                    potential_header = df.iloc[0]
                    if not potential_header.isna().all() and potential_header.notna().all():
                        df.columns = df.iloc[0]
                        df = df.iloc[1:].reset_index(drop=True)
                
                # Clean the DataFrame
                df = df.replace('', None)
                df = df.dropna(how='all', axis=1)  # Drop empty columns
                df = df.dropna(how='all', axis=0)  # Drop empty rows
                
                # Convert to dict for JSON serialization
                processed_table = {
                    'table_id': table_idx,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'headers': df.columns.tolist(),
                    'data': df.values.tolist()
                }
                processed_tables.append(processed_table)
        
        return processed_tables

    def extract_pdf_content(self, pdf_path):
        """Extract content, tables, and metadata from a PDF file."""
        try:
            # Extract basic content and metadata using PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = self.extract_pdf_metadata(pdf_reader)
            
            # Extract tables and detailed content using pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                content = []
                total_pages = len(pdf.pages)
                
                for page_num, page in tqdm(
                    enumerate(pdf.pages, 1),
                    total=total_pages,
                    desc=f"Processing pages in {pdf_path.name}",
                    leave=False
                ):
                    # Extract text
                    text = page.extract_text()
                    
                    # Extract tables
                    tables = self.extract_tables_from_page(page)
                    
                    if text.strip() or tables:  # Only add non-empty pages
                        page_content = {
                            'page_number': page_num,
                            'text': text.strip() if text else '',
                            'tables': tables
                        }
                        content.append(page_content)

                # Create JSON structure
                result = {
                    'metadata': {
                        'filename': pdf_path.name,
                        'file_path': str(pdf_path),
                        'file_hash': self.generate_file_hash(pdf_path),
                        'extraction_date': datetime.now().isoformat(),
                        'pdf_metadata': metadata,
                        'table_summary': {
                            'total_tables': sum(len(page.get('tables', [])) for page in content),
                            'pages_with_tables': sum(1 for page in content if page.get('tables', []))
                        }
                    },
                    'content': content
                }

                return result

        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return None

    def process_pdfs(self):
        """Process all PDFs in the input directory."""
        results = []
        
        # Get list of PDF files
        pdf_files = list(self.input_dir.glob('*.pdf'))
        
        # Process each PDF file with progress bar
        for pdf_path in tqdm(pdf_files, desc="Processing PDF files", unit="file"):
            result = self.extract_pdf_content(pdf_path)
            if result:
                # Save individual PDF result
                output_file = self.output_dir / f"{pdf_path.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                results.append(result)
        
        # Save collection summary
        summary = {
            'collection_metadata': {
                'total_documents': len(results),
                'processing_date': datetime.now().isoformat(),
                'input_directory': str(self.input_dir),
                'output_directory': str(self.output_dir),
                'total_tables_across_documents': sum(
                    doc['metadata']['table_summary']['total_tables'] 
                    for doc in results
                )
            },
            'documents': results
        }
        
        with open(self.output_dir / 'collection_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = PDFParser()
    parser.process_pdfs()