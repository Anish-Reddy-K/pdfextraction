"""
Table Visualization Script
-------------------------
Displays tables extracted from PDF documents, showing table content with context and citations.
Works with JSON files created by extract_basic.py.

Usage:
    python table.py                     # Process all documents with default filters
    python table.py Benzene            # Process specific document
    
Filters available:
    - Column filter (tables with >1 column)
    - Row filter (tables with >5 rows)
    - Custom filters can be added in main()
"""

# Setup
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table as RichTable
from rich import box
from rich.panel import Panel
from rich.text import Text
from tqdm import tqdm
from typing import Optional, Callable, Dict, Any, List

def load_document(processed_dir: Path, filename: str) -> Dict[str, Any]:
    """Load document data from JSON file."""
    if filename == 'all':
        # For 'all', we'll load each JSON file in the directory
        json_files = list(processed_dir.glob('*.json'))
        all_docs = []
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_docs.append(json.load(f))
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        return all_docs
    else:
        # For specific document
        file_path = processed_dir / f"{filename}.json"
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.load(f)]

def create_rich_table(table_data: Dict[str, Any]) -> RichTable:
    """Create a rich table from the table data."""
    metadata = table_data['metadata']
    
    # Create Rich table with row separators
    rich_table = RichTable(box=box.DOUBLE_EDGE, show_header=False, show_lines=True)
    
    # Add columns
    for _ in range(metadata['columns']):
        rich_table.add_column("")
    
    # Add rows
    for row in table_data['data']:
        rich_table.add_row(*[str(cell) if cell is not None else '' for cell in row])
    
    return rich_table

def create_citation_text(table_data: Dict[str, Any], filename: str) -> Text:
    """Create citation text for a table."""
    metadata = table_data['metadata']
    citation = Text()
    
    # Add source
    citation.append(f"Source: {filename}\n", style="bold blue")
    citation.append(f"Page: {metadata['page_number']}\n", style="blue")
    
    # Add headings if available
    if metadata['context_headings']:
        citation.append("\nHeadings:\n", style="bold yellow")
        for heading in metadata['context_headings']:
            citation.append(f"• {heading}\n", style="yellow")
    
    # Add surrounding text
    if metadata['text_before']:
        citation.append("\nText before table:\n", style="bold green")
        citation.append(f'"{metadata["text_before"]}"\n', style="green")
    
    if metadata['text_after']:
        citation.append("\nText after table:\n", style="bold red")
        citation.append(f'"{metadata["text_after"]}"\n', style="red")
    
    return citation

def display_table(console: Console, table_data: Dict[str, Any], filename: str) -> None:
    """Display a single table with its citation."""
    rich_table = create_rich_table(table_data)
    citation = create_citation_text(table_data, filename)
    
    console.print("\n" + "="*80 + "\n")
    console.print(Panel(citation, title="Table Citation", border_style="blue"))
    console.print(rich_table)
    console.print("\n")

def process_tables(
    processed_dir: str = 'processed',
    filename: str = 'all',
    table_filter: Optional[Callable[[Dict[str, Any]], bool]] = None
) -> None:
    """Process and display tables from document(s)."""
    console = Console()
    processed_path = Path(processed_dir)
    
    try:
        # Load document(s)
        documents = load_document(processed_path, filename)
        
        # Process each document
        for doc in tqdm(documents, desc="Processing documents"):
            doc_filename = doc['metadata']['filename']
            console.print(f"\n[bold cyan]Document: {doc_filename}[/bold cyan]")
            
            # Process each page
            for page in doc['content']:
                for table in page['tables']:
                    if table_filter is None or table_filter(table):
                        display_table(console, table, doc_filename)
                        
    except FileNotFoundError:
        console.print(f"[bold red]Error: Could not find document file for {filename}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise

def main():
    # Filters
    column_filter = lambda table: table['metadata']['columns'] > 1
    row_filter = lambda table: table['metadata']['rows'] > 5

    print("Processing tables and showing results...")
    
    # Process all documents
    # process_tables(filename='all', table_filter=column_filter)  # column_filter
    # process_tables(filename='all', table_filter=row_filter)  # row_filter
    
    # Process specific document
    process_tables(filename='Benzene', table_filter=column_filter)

if __name__ == "__main__":
    main()