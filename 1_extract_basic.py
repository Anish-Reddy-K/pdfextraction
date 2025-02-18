import os
from datetime import datetime
from pathlib import Path
import json
import PyPDF2
import pdfplumber
import hashlib
import pandas as pd
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self, input_dir='uploads', output_dir='processed'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_file_hash(self, file_path):
        """Generate SHA-256 hash of a file for tracking changes."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def extract_pdf_metadata(self, pdf_reader):
        """Extract metadata from a PDF document."""
        metadata = pdf_reader.metadata or {}
        return {
            'title': metadata.get('/Title', ''),
            'author': metadata.get('/Author', ''),
            'subject': metadata.get('/Subject', ''),
            'creator': metadata.get('/Creator', ''),
            'creation_date': metadata.get('/CreationDate', ''),
            'page_count': len(pdf_reader.pages)
        }

    def extract_table_context(self, page, table_bbox, text_blocks):
        """Extract context around a table (headings, text before/after)."""
        table_top, table_bottom = table_bbox[1], table_bbox[3]
        context = {'potential_headings': [], 'text_before': [], 'text_after': []}

        for block in text_blocks:
            block_bbox = block.get('bbox')
            if not block_bbox:
                continue

            block_bottom, block_top = block_bbox[3], block_bbox[1]
            text = block.get('text', '').strip()

            if not text:
                continue

            # Text blocks just above the table (potential headings)
            if block_bottom < table_top and block_bottom >= table_top - 50:
                context['potential_headings'].append({
                    'text': text,
                    'distance_from_table': table_top - block_bottom
                })
            # Text before the table
            elif block_bottom < table_top:
                context['text_before'].append(text)
            # Text after the table
            elif block_top > table_bottom:
                context['text_after'].append(text)

        # Keep only the 2 closest headings
        context['potential_headings'] = sorted(
            context['potential_headings'],
            key=lambda x: x['distance_from_table']
        )[:2]

        return {
            'potential_headings': [h['text'] for h in context['potential_headings']],
            'text_before': ' '.join(context['text_before'][-2:]),
            'text_after': ' '.join(context['text_after'][:2])
        }

    def extract_tables_from_page(self, page, page_num):
        """Extract tables and their context from a single page."""
        tables = page.extract_tables()
        processed_tables = []

        if tables:
            text_blocks = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)

            for table_idx, table in enumerate(tables):
                table_bbox = page.find_tables()[table_idx].bbox
                df = pd.DataFrame(table).replace('', None).dropna(how='all', axis=1).dropna(how='all', axis=0)
                df.columns = [f'Column_{i+1}' for i in range(len(df.columns))]

                context = self.extract_table_context(page, table_bbox, text_blocks)

                processed_tables.append({
                    'table_id': table_idx + 1,
                    'position': {'bbox': list(table_bbox), 'page_number': page_num},
                    'context': context,
                    'structure': {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'headers': df.columns.tolist(),
                        'data': df.values.tolist()
                    }
                })

        return processed_tables

    def extract_pdf_content(self, pdf_path):
        """Extract content, tables, and metadata from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = self.extract_pdf_metadata(pdf_reader)

            with pdfplumber.open(pdf_path) as pdf:
                content = []
                table_contexts = []

                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    tables = self.extract_tables_from_page(page, page_num)

                    if tables:
                        table_contexts.extend(tables)

                    if text.strip() or tables:
                        content.append({
                            'page_number': page_num,
                            'text': text.strip() if text else '',
                            'tables': tables
                        })

                result = {
                    'metadata': {
                        'filename': pdf_path.name,
                        'file_path': str(pdf_path),
                        'file_hash': self.generate_file_hash(pdf_path),
                        'extraction_date': datetime.now().isoformat(),
                        'pdf_metadata': metadata,
                        'table_summary': {
                            'total_tables': len(table_contexts),
                            'pages_with_tables': sum(1 for page in content if page.get('tables', []))
                        }
                    },
                    'content': content
                }

                table_context_data = {
                    'file_metadata': {
                        'filename': pdf_path.name,
                        'extraction_date': datetime.now().isoformat(),
                        'total_tables': len(table_contexts)
                    },
                    'tables': table_contexts
                }

                # Save table contexts to a separate file
                table_context_file = self.output_dir / f"{pdf_path.stem}_table_contexts.json"
                with open(table_context_file, 'w', encoding='utf-8') as f:
                    json.dump(table_context_data, f, ensure_ascii=False, indent=2)

                return result, table_context_data

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return None, None

    def process_pdfs(self):
        """Process all PDFs in the input directory."""
        results = []
        all_table_contexts = []

        pdf_files = list(self.input_dir.glob('*.pdf'))
        for pdf_path in tqdm(pdf_files, desc="Processing PDF files", unit="file"):
            result, table_contexts = self.extract_pdf_content(pdf_path)
            if result:
                output_file = self.output_dir / f"{pdf_path.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                results.append(result)
                all_table_contexts.append(table_contexts)

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

        all_contexts = {
            'collection_metadata': {
                'total_documents': len(results),
                'processing_date': datetime.now().isoformat(),
                'total_tables': sum(ctx['file_metadata']['total_tables'] for ctx in all_table_contexts if ctx)
            },
            'documents': all_table_contexts
        }

        with open(self.output_dir / 'collection_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with open(self.output_dir / 'all_table_contexts.json', 'w', encoding='utf-8') as f:
            json.dump(all_contexts, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = PDFParser()
    parser.process_pdfs()