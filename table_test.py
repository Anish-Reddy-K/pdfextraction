from pathlib import Path
import json
from rich.console import Console
from rich.table import Table as RichTable
from rich import box
from rich.panel import Panel
from rich.text import Text
from tqdm import tqdm
from typing import Optional, Callable, Dict, Any

def load_table_contexts(processed_dir: Path, filename: str) -> Dict[str, Any]:
    """Load table contexts from a specific file or all files."""
    if filename == 'all':
        file_path = processed_dir / 'all_table_contexts.json'
    else:
        file_path = processed_dir / f"{filename}_table_contexts.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_rich_table(table_data: Dict[str, Any]) -> RichTable:
    """Create a rich table from the table data."""
    # Create Rich table with row separators
    rich_table = RichTable(box=box.DOUBLE_EDGE, show_header=False, show_lines=True)
    
    # Add columns
    metadata = table_data['metadata']
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
    # Create the table
    rich_table = create_rich_table(table_data)
    
    # Create the citation
    citation = create_citation_text(table_data, filename)
    
    # Display everything
    console.print("\n" + "="*80 + "\n")
    console.print(Panel(citation, title="Table Citation", border_style="blue"))
    console.print(rich_table)
    console.print("\n")

def reconstruct_tables(
    processed_dir: str = 'processed',
    filename: str = 'all',
    table_filter: Optional[Callable[[Dict[str, Any]], bool]] = None
) -> None:
    """Reconstruct and display tables from JSON data."""
    console = Console()
    processed_path = Path(processed_dir)
    
    try:
        # Load table contexts
        data = load_table_contexts(processed_path, filename)
        
        if filename == 'all':
            # Process all documents
            for doc in tqdm(data['documents'], desc="Processing documents"):
                if doc:  # Check if document data exists
                    doc_filename = doc['file_metadata']['filename']
                    console.print(f"\n[bold cyan]Document: {doc_filename}[/bold cyan]")
                    
                    for table in doc['tables']:
                        if table_filter is None or table_filter(table):
                            display_table(console, table, doc_filename)
        else:
            # Process single document
            for table in tqdm(data['tables'], desc="Processing tables"):
                if table_filter is None or table_filter(table):
                    display_table(console, table, filename)
                    
    except FileNotFoundError:
        console.print(f"[bold red]Error: Could not find table context file for {filename}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise  # Re-raise for debugging

def main():
    # Example usage:
    print("Reconstructing all tables from all documents...")
    
    # Define example filters
    column_filter = lambda table: table['metadata']['columns'] > 1
    row_filter = lambda table: table['metadata']['rows'] > 5
    
    # 1. Reconstruct all tables from all documents with column filter
    # reconstruct_tables(filename='all', table_filter=column_filter)
    
    # 2. Reconstruct tables from a specific document
    reconstruct_tables(filename='Benzene', table_filter=column_filter)
    
    # 3. Reconstruct tables with row filter
    # reconstruct_tables(filename='all', table_filter=row_filter)

if __name__ == "__main__":
    main()