import re
import json
import pandas as pd
from typing import List, Dict, Any
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict


def parse_markdown_tables(markdown: str) -> List[Dict[str, Any]]:
    """Parse all GitHub-style markdown tables from a markdown string."""
    lines = markdown.splitlines()
    tables = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect a potential table header row
        if line.startswith("|") and line.endswith("|") and ("|" in line[1:-1]):
            header_line = line
            if i + 1 < len(lines):
                separator_line = lines[i + 1].strip()
            else:
                i += 1
                continue
            
            # Check if next line is a separator (--- style)
            if separator_line.startswith("|") and separator_line.endswith("|") and re.search(r"-", separator_line):
                # We found a table: collect lines until a non-table line
                table_lines = [header_line, separator_line]
                j = i + 2
                while j < len(lines):
                    l = lines[j].strip()
                    if l.startswith("|") and l.endswith("|") and ("|" in l[1:-1]):
                        table_lines.append(l)
                        j += 1
                    else:
                        break
                
                # Parse this table
                table = _parse_single_markdown_table(table_lines)
                if table and table["rows"]:  # Only add non-empty tables
                    tables.append(table)
                
                # Skip past this table
                i = j
                continue
        
        i += 1
    
    return tables


def _split_markdown_row(row: str) -> List[str]:
    """Split a markdown table row into cells."""
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    cells = [c.strip() for c in row.split("|")]
    return cells


def _parse_single_markdown_table(lines: List[str]) -> Dict[str, Any]:
    """Parse a single markdown table into headers/rows."""
    if len(lines) < 3:  # Need at least header, separator, and one data row
        return None
    
    header_cells = _split_markdown_row(lines[0])
    
    data_rows = []
    for row_line in lines[2:]:
        cells = _split_markdown_row(row_line)
        # Normalize row length
        if len(cells) < len(header_cells):
            cells.extend([""] * (len(header_cells) - len(cells)))
        elif len(cells) > len(header_cells):
            cells = cells[:len(header_cells)]
        data_rows.append(cells)
    
    return {
        "headers": header_cells,
        "rows": data_rows,
    }


def clean_cell_text(text: str) -> str:
    """Clean up cell text: remove <br>, normalize whitespace."""
    text = text.replace("<br>", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_all_tables_from_pdf(pdf_path: str):
    """
    Extract ALL tables from PDF using marker-pdf.
    Returns detailed cell-level information for each table.
    """
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    
    rendered = converter(pdf_path)
    markdown = rendered.markdown
    
    # Debug: print the full markdown to see what marker detected
    print("=" * 80)
    print("FULL MARKDOWN OUTPUT:")
    print("=" * 80)
    print(markdown)
    print("=" * 80)
    
    # Parse all markdown tables
    raw_tables = parse_markdown_tables(markdown)
    
    print(f"\nFound {len(raw_tables)} tables in markdown")
    
    # Convert to structured format with cell-level details
    structured_tables = []
    for t_idx, t in enumerate(raw_tables):
        table_info = {
            "index": t_idx,
            "num_columns": len(t["headers"]),
            "num_rows": len(t["rows"]),
            "headers": [],
            "cells": [],
        }
        
        # Clean header cells
        headers = [clean_cell_text(h) for h in t["headers"]]
        table_info["headers"] = headers
        
        # Data rows with cell-level info
        for r_idx, row in enumerate(t["rows"]):
            for c_idx, cell in enumerate(row):
                cell_text = clean_cell_text(cell)
                table_info["cells"].append({
                    "row": r_idx,
                    "col": c_idx,
                    "header": headers[c_idx] if c_idx < len(headers) else None,
                    "text": cell_text,
                })
        
        structured_tables.append(table_info)
    
    return structured_tables, markdown


def tables_to_dataframes(structured_tables):
    """Convert structured tables to pandas DataFrames."""
    dfs = []
    
    for t in structured_tables:
        headers = t["headers"]
        rows_by_index = {}
        
        for c in t["cells"]:
            r = c["row"]
            col_idx = c["col"]
            if r not in rows_by_index:
                rows_by_index[r] = {h: "" for h in headers}
            if col_idx < len(headers):
                header = headers[col_idx]
                rows_by_index[r][header] = c["text"]
        
        ordered_rows = [rows_by_index[r] for r in sorted(rows_by_index)]
        df = pd.DataFrame(ordered_rows, columns=headers)
        dfs.append({"index": t["index"], "df": df, "shape": df.shape})
    
    return dfs


def extract_table_regions_from_metadata(pdf_path: str):
    """
    Check marker's metadata to see what table regions were detected.
    This helps debug why some tables might be missing.
    """
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    
    rendered = converter(pdf_path)
    
    print("\n" + "=" * 80)
    print("MARKER METADATA - TABLE DETECTION INFO:")
    print("=" * 80)
    
    if hasattr(rendered, 'metadata') and rendered.metadata:
        metadata = rendered.metadata
        
        if 'page_stats' in metadata:
            for page_idx, page_stat in enumerate(metadata['page_stats']):
                print(f"\nPage {page_idx}:")
                print(f"  Text extraction method: {page_stat.get('text_extraction_method', 'N/A')}")
                
                if 'block_counts' in page_stat:
                    print(f"  Block counts:")
                    for block_type, count in page_stat['block_counts']:
                        print(f"    {block_type}: {count}")
                        
                if 'block_metadata' in page_stat:
                    print(f"  Block metadata: {page_stat['block_metadata']}")
    
    return rendered


if __name__ == "__main__":
    pdf_path = "input/sample-3.pdf"
    
    print("Extracting tables from sample-3.pdf using marker-py...\n")
    
    # Step 1: Extract metadata to see what marker detected
    rendered = extract_table_regions_from_metadata(pdf_path)
    
    # Step 2: Extract all tables
    tables, full_markdown = extract_all_tables_from_pdf(pdf_path)
    
    print(f"\n{'=' * 80}")
    print(f"EXTRACTED {len(tables)} TABLES:")
    print("=" * 80)
    
    # Display each table
    for t in tables:
        print(f"\n--- Table {t['index']} ---")
        print(f"Shape: {t['num_rows']} rows × {t['num_columns']} columns")
        print(f"Headers: {t['headers']}")
        print(f"\nFirst 10 cells:")
        for c in t['cells'][:10]:
            print(f"  Row {c['row']}, Col {c['col']} ({c['header']}): {c['text']}")
    
    # Step 3: Save to JSON
    with open('marker_tables_output.json', 'w', encoding='utf-8') as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved detailed table data to marker_tables_output.json")
    
    # Step 4: Convert to DataFrames and save as CSV
    dfs = tables_to_dataframes(tables)
    
    print(f"\n{'=' * 80}")
    print("DATAFRAMES:")
    print("=" * 80)
    
    for t in dfs:
        print(f"\n--- Table {t['index']} (shape: {t['shape']}) ---")
        print(t['df'])
        
        # Save to CSV
        csv_filename = f"marker_table_{t['index']}.csv"
        t['df'].to_csv(csv_filename, index=False)
        print(f"✓ Saved to {csv_filename}")
    
    # Step 5: Save full markdown for inspection
    with open('marker_full_output.md', 'w', encoding='utf-8') as f:
        f.write(full_markdown)
    print(f"\n✓ Saved full markdown to marker_full_output.md")