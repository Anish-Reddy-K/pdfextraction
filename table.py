#table.py

import json
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table as RichTable
from rich import box
from rich.panel import Panel
from rich.text import Text
from tqdm import tqdm

class TableReconstructor:
    def __init__(self, processed_dir='processed'):
        self.processed_dir = Path(processed_dir)
        self.console = Console()
        
    def load_table_contexts(self, filename):
        """Load table contexts from a specific file or all files."""
        if filename == 'all':
            file_path = self.processed_dir / 'all_table_contexts.json'
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            file_path = self.processed_dir / f"{filename}_table_contexts.json"
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)

    def create_rich_table(self, table_data):
        """Create a rich table from the table data."""
        headers = table_data['structure']['headers']
        data = table_data['structure']['data']
        
        # Create Rich table with row separators
        rich_table = RichTable(box=box.DOUBLE_EDGE, show_header=False, show_lines=True)
        
        # Add columns
        for _ in range(len(table_data['structure']['headers'])):
            rich_table.add_column("")
            
        # Add rows
        for row in data:
            rich_table.add_row(*[str(cell) if cell is not None else '' for cell in row])
            
        return rich_table

    def create_citation_text(self, table_data, filename):
        """Create citation text for a table."""
        context = table_data['context']
        position = table_data['position']
        
        citation = Text()
        
        # Add source
        citation.append(f"Source: {filename}\n", style="bold blue")
        citation.append(f"Page: {position['page_number']}\n", style="blue")
        
        # Add headings if available
        if context['potential_headings']:
            citation.append("\nHeadings:\n", style="bold yellow")
            for heading in context['potential_headings']:
                citation.append(f"• {heading}\n", style="yellow")
        
        # Add surrounding text
        if context['text_before']:
            citation.append("\nText before table:\n", style="bold green")
            citation.append(f'"{context["text_before"]}"\n', style="green")
        
        if context['text_after']:
            citation.append("\nText after table:\n", style="bold red")
            citation.append(f'"{context["text_after"]}"\n', style="red")
            
        return citation

    def display_table(self, table_data, filename):
        """Display a single table with its citation."""
        # Create the table
        rich_table = self.create_rich_table(table_data)
        
        # Create the citation
        citation = self.create_citation_text(table_data, filename)
        
        # Display everything
        self.console.print("\n" + "="*80 + "\n")
        self.console.print(Panel(citation, title="Table Citation", border_style="blue"))
        self.console.print(rich_table)
        self.console.print("\n")

    def reconstruct_tables(self, filename='all', table_filter=None):
        """Reconstruct and display tables from JSON data."""
        try:
            # Load table contexts
            data = self.load_table_contexts(filename)
            
            if filename == 'all':
                # Process all documents
                for doc in tqdm(data['documents'], desc="Processing documents"):
                    if doc:  # Check if document data exists
                        doc_filename = doc['file_metadata']['filename']
                        self.console.print(f"\n[bold cyan]Document: {doc_filename}[/bold cyan]")
                        
                        for table in doc['tables']:
                            if table_filter is None or table_filter(table):
                                self.display_table(table, doc_filename)
            else:
                # Process single document
                for table in tqdm(data['tables'], desc="Processing tables"):
                    if table_filter is None or table_filter(table):
                        self.display_table(table, filename)
                        
        except FileNotFoundError:
            self.console.print(f"[bold red]Error: Could not find table context file for {filename}[/bold red]")
        except Exception as e:
            self.console.print(f"[bold red]Error: {str(e)}[/bold red]")

def main():
    reconstructor = TableReconstructor()
    
    # Example usage:
    
    # 1. Reconstruct all tables from all documents
    print("Reconstructing all tables from all documents...")
    filter_func = lambda table: table['structure']['columns'] > 1 
    #reconstructor.reconstruct_tables('all', table_filter=filter_func)
    reconstructor.reconstruct_tables('Benzene', table_filter=filter_func)


    # 2. Reconstruct tables from a specific document
    # print("\nReconstructing tables from specific document...")
    # reconstructor.reconstruct_tables('example_document')
    
    # 3. Reconstruct tables with specific criteria
    # print("\nReconstructing tables with specific criteria...")
    # filter_func = lambda table: table['structure']['rows'] > 5  # Only tables with more than 5 rows
    # reconstructor.reconstruct_tables('all', table_filter=filter_func)
    # print("\nReconstructing tables with specific criteria...")
    # reconstructor.reconstruct_tables('asphalt_mc-70-2022', table_filter=filter_func)

if __name__ == "__main__":
    main()