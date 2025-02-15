#!/usr/bin/env python3
"""
PDF Processing Script
Extracts content from PDFs in 'uploads' folder and stores structured data in JSON format.
Uses only open-source libraries suitable for commercial use.
"""

import os
import json
from datetime import datetime
from pathlib import Path
import hashlib
from typing import Dict, List, Optional

# PyPDF2 is open source (BSD License)
from PyPDF2 import PdfReader
# python-magic is open source (MIT License)
import magic
# nltk is open source (Apache License 2.0)
import nltk
from nltk.tokenize import sent_tokenize

def ensure_nltk_resources():
    """Ensure all required NLTK resources are downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        #print("Downloading required NLTK resources...")
        nltk.download('punkt', quiet=True)
        #print("Download complete.")

class PDFProcessor:
    def __init__(self, upload_dir: str = 'uploads', output_dir: str = 'processed_data'):
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        ensure_nltk_resources()
        
    def get_file_metadata(self, file_path: Path) -> Dict:
        """Extract metadata from the PDF file."""
        stats = file_path.stat()
        mime = magic.Magic(mime=True)
        
        return {
            "filename": file_path.name,
            "file_size": stats.st_size,
            "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "mime_type": mime.from_file(str(file_path)),
            "file_hash": self.calculate_file_hash(file_path)
        }
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of the file for verification."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def extract_pdf_content(self, pdf_path: Path) -> Dict:
        """Extract content and metadata from PDF."""
        try:
            reader = PdfReader(str(pdf_path))
            
            # Extract PDF metadata
            pdf_info = reader.metadata
            if pdf_info:
                pdf_metadata = {
                    "title": pdf_info.get('/Title', ''),
                    "author": pdf_info.get('/Author', ''),
                    "subject": pdf_info.get('/Subject', ''),
                    "creator": pdf_info.get('/Creator', ''),
                    "creation_date": pdf_info.get('/CreationDate', ''),
                }
            else:
                pdf_metadata = {}
            
            # Extract text content page by page
            pages_content = []
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                
                # Split into sentences for better processing by LLMs
                sentences = []
                if text.strip():  # Only process if there's actual text
                    try:
                        sentences = sent_tokenize(text)
                    except Exception as e:
                        print(f"Warning: Sentence tokenization failed: {str(e)}")
                        sentences = [text]  # Use the whole text as a single sentence if tokenization fails
                
                page_data = {
                    "page_number": page_num,
                    "content": text,
                    "sentences": sentences,
                    "word_count": len(text.split()),
                }
                pages_content.append(page_data)
            
            return {
                "metadata": pdf_metadata,
                "pages": pages_content,
                "total_pages": len(reader.pages)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "metadata": {},
                "pages": [],
                "total_pages": 0
            }
    
    def process_pdfs(self) -> None:
        """Process all PDFs in the upload directory."""
        for pdf_file in self.upload_dir.glob('*.pdf'):
            try:
                # Create the JSON structure
                json_data = {
                    "processing_metadata": {
                        "processed_at": datetime.now().isoformat(),
                        "processor_version": "1.0.0",
                        "python_version": ".".join(map(str, os.sys.version_info[:3])),
                    },
                    "file_metadata": self.get_file_metadata(pdf_file),
                    "content": self.extract_pdf_content(pdf_file)
                }
                
                # Save to JSON file
                output_file = self.output_dir / f"{pdf_file.stem}_processed.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                print(f"Successfully processed: {pdf_file.name}")
                
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {str(e)}")

if __name__ == "__main__":
    processor = PDFProcessor()
    processor.process_pdfs()