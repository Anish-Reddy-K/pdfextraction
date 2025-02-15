import os
from datetime import datetime
from pathlib import Path
import json
import PyPDF2
import hashlib

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

    def extract_pdf_content(self, pdf_path):
        """Extract content and metadata from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                # Create PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = self.extract_pdf_metadata(pdf_reader)
                
                # Extract text content from each page
                content = []
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        content.append({
                            'page_number': page_num,
                            'text': text.strip()
                        })

                # Create JSON structure
                result = {
                    'metadata': {
                        'filename': pdf_path.name,
                        'file_path': str(pdf_path),
                        'file_hash': self.generate_file_hash(pdf_path),
                        'extraction_date': datetime.now().isoformat(),
                        'pdf_metadata': metadata
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
        
        # Process each PDF file
        for pdf_path in self.input_dir.glob('*.pdf'):
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
                'output_directory': str(self.output_dir)
            },
            'documents': results
        }
        
        with open(self.output_dir / 'collection_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = PDFParser()
    parser.process_pdfs()