import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from PIL import Image
import pytesseract


def parse_markdown_tables(markdown: str) -> List[Dict[str, Any]]:
    """Parse all GitHub-style markdown tables from a markdown string."""
    lines = markdown.splitlines()
    tables = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith("|") and line.endswith("|") and ("|" in line[1:-1]):
            header_line = line
            if i + 1 < len(lines):
                separator_line = lines[i + 1].strip()
            else:
                i += 1
                continue
            
            if separator_line.startswith("|") and separator_line.endswith("|") and re.search(r"-", separator_line):
                table_lines = [header_line, separator_line]
                j = i + 2
                while j < len(lines):
                    l = lines[j].strip()
                    if l.startswith("|") and l.endswith("|") and ("|" in l[1:-1]):
                        table_lines.append(l)
                        j += 1
                    else:
                        break
                
                table = _parse_single_markdown_table(table_lines)
                if table and table["rows"]:
                    tables.append(table)
                
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
    if len(lines) < 3:
        return None
    
    header_cells = _split_markdown_row(lines[0])
    
    data_rows = []
    for row_line in lines[2:]:
        cells = _split_markdown_row(row_line)
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


def extract_figure_references(markdown: str) -> List[str]:
    """Extract all figure image references from markdown."""
    pattern = r'!\[\]\(([^)]+)\)'
    matches = re.findall(pattern, markdown)
    return matches


def ocr_table_from_image(image_path: str) -> str:
    """
    Use OCR to extract text from an image that contains a table.
    Returns the raw OCR text.
    """
    try:
        img = Image.open(image_path)
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error OCR'ing image {image_path}: {e}")
        return ""


def parse_ocr_table_text(ocr_text: str) -> Dict[str, Any]:
    """
    Parse OCR text that represents a table.
    This is a heuristic parser - adjust based on your table format.
    """
    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]
    
    if not lines:
        return None
    
    # Try to detect header row (usually first line)
    rows = []
    for line in lines:
        # Split by 2+ spaces or tabs
        cells = re.split(r'\s{2,}|\t+', line)
        cells = [c.strip() for c in cells if c.strip()]
        if cells:
            rows.append(cells)
    
    if not rows:
        return None
    
    # Assume first row is header
    headers = rows[0]
    data_rows = rows[1:]
    
    # Normalize row lengths
    max_cols = len(headers)
    normalized_rows = []
    for row in data_rows:
        if len(row) < max_cols:
            row.extend([""] * (max_cols - len(row)))
        elif len(row) > max_cols:
            row = row[:max_cols]
        normalized_rows.append(row)
    
    return {
        "headers": headers,
        "rows": normalized_rows,
    }


def extract_all_tables_and_figures(pdf_path: str, output_dir: str = "."):
    """
    Extract ALL tables from PDF including those detected as figures.
    """
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    
    rendered = converter(pdf_path)
    markdown = rendered.markdown
    
    # Save images if they exist (they are PIL Image objects)
    if hasattr(rendered, 'images') and rendered.images:
        for img_name, img_obj in rendered.images.items():
            img_path = Path(output_dir) / img_name
            # Save PIL Image object
            if isinstance(img_obj, Image.Image):
                img_obj.save(img_path)
                print(f"✓ Saved figure: {img_path}")
            else:
                # If it's bytes, write directly
                with open(img_path, 'wb') as f:
                    f.write(img_obj)
                print(f"✓ Saved figure: {img_path}")
    
    print("\n" + "=" * 80)
    print("FULL MARKDOWN OUTPUT:")
    print("=" * 80)
    print(markdown)
    print("=" * 80)
    
    # Extract markdown tables
    markdown_tables = parse_markdown_tables(markdown)
    print(f"\nFound {len(markdown_tables)} markdown tables")
    
    # Extract figure references
    figure_refs = extract_figure_references(markdown)
    print(f"Found {len(figure_refs)} figures")
    
    all_tables = []
    
    # Process figures (potential tables)
    for fig_idx, fig_ref in enumerate(figure_refs):
        print(f"\nProcessing figure: {fig_ref}")
        fig_path = Path(output_dir) / fig_ref
        
        if fig_path.exists():
            # Try OCR on the figure
            ocr_text = ocr_table_from_image(str(fig_path))
            print(f"OCR text from {fig_ref}:")
            print(ocr_text)
            print("-" * 40)
            
            # Try to parse as table
            parsed_table = parse_ocr_table_text(ocr_text)
            if parsed_table:
                table_info = {
                    "index": len(all_tables),
                    "source": "figure_ocr",
                    "figure_path": fig_ref,
                    "num_columns": len(parsed_table["headers"]),
                    "num_rows": len(parsed_table["rows"]),
                    "headers": parsed_table["headers"],
                    "cells": [],
                }
                
                # Build cell-level structure
                for r_idx, row in enumerate(parsed_table["rows"]):
                    for c_idx, cell in enumerate(row):
                        table_info["cells"].append({
                            "row": r_idx,
                            "col": c_idx,
                            "header": parsed_table["headers"][c_idx] if c_idx < len(parsed_table["headers"]) else None,
                            "text": clean_cell_text(cell),
                        })
                
                all_tables.append(table_info)
                print(f"✓ Extracted table from figure with {len(parsed_table['rows'])} rows")
            else:
                print(f"⚠ Could not parse table structure from OCR text")
        else:
            print(f"⚠ Figure file not found: {fig_path}")
    
    # Process markdown tables
    for t_idx, t in enumerate(markdown_tables):
        table_info = {
            "index": len(all_tables),
            "source": "markdown",
            "num_columns": len(t["headers"]),
            "num_rows": len(t["rows"]),
            "headers": [clean_cell_text(h) for h in t["headers"]],
            "cells": [],
        }
        
        headers = table_info["headers"]
        for r_idx, row in enumerate(t["rows"]):
            for c_idx, cell in enumerate(row):
                cell_text = clean_cell_text(cell)
                table_info["cells"].append({
                    "row": r_idx,
                    "col": c_idx,
                    "header": headers[c_idx] if c_idx < len(headers) else None,
                    "text": cell_text,
                })
        
        all_tables.append(table_info)
    
    return all_tables, markdown


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
        dfs.append({
            "index": t["index"],
            "source": t.get("source", "unknown"),
            "df": df,
            "shape": df.shape
        })
    
    return dfs


if __name__ == "__main__":
    pdf_path = "input/sample-4.pdf"
    output_dir = "."
    
    print("Extracting ALL tables (including figures) from sample-4.pdf using marker-py...\n")
    
    # Extract all tables and figures
    tables, full_markdown = extract_all_tables_and_figures(pdf_path, output_dir)
    
    print(f"\n{'=' * 80}")
    print(f"EXTRACTED {len(tables)} TOTAL TABLES:")
    print("=" * 80)
    
    # Display each table
    for t in tables:
        print(f"\n--- Table {t['index']} (source: {t['source']}) ---")
        print(f"Shape: {t['num_rows']} rows × {t['num_columns']} columns")
        print(f"Headers: {t['headers']}")
        print(f"\nFirst 10 cells:")
        for c in t['cells'][:10]:
            print(f"  Row {c['row']}, Col {c['col']} ({c['header']}): {c['text']}")
    
    # Save to JSON
    with open('all_tables_output.json', 'w', encoding='utf-8') as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved detailed table data to all_tables_output.json")
    
    # Convert to DataFrames
    dfs = tables_to_dataframes(tables)
    
    print(f"\n{'=' * 80}")
    print("DATAFRAMES:")
    print("=" * 80)
    
    for t in dfs:
        print(f"\n--- Table {t['index']} (source: {t['source']}, shape: {t['shape']}) ---")
        print(t['df'])
        
        # Save to CSV
        csv_filename = f"table_{t['index']}_{t['source']}.csv"
        t['df'].to_csv(csv_filename, index=False)
        print(f"✓ Saved to {csv_filename}")