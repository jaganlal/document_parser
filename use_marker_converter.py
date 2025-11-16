"""
Improved Table Extraction System
Handles BOTH vertical key-value tables and standard horizontal tables automatically.
"""

import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from PIL import Image
import pytesseract


def parse_markdown_tables_with_context(markdown: str) -> List[Dict[str, Any]]:
    """
    Parse all GitHub-style markdown tables from a markdown string.
    Also captures table titles/captions that appear before the table.
    """
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
                # Found a table! Now look backwards for title/caption
                title = _extract_table_title(lines, i)
                
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
                    table["title"] = title
                    table["line_start"] = i
                    table["line_end"] = j - 1
                    tables.append(table)
                
                i = j
                continue
        
        i += 1
    
    return tables


def _extract_table_title(lines: List[str], table_start_line: int) -> str:
    """Extract table title by looking at lines before the table."""
    title_candidates = []
    lookback = min(5, table_start_line)
    
    for i in range(table_start_line - 1, table_start_line - lookback - 1, -1):
        if i < 0:
            break
        
        line = lines[i].strip()
        
        if not line or line.startswith("![]") or re.match(r'^[-=*_]{3,}$', line):
            continue
        
        # Check for markdown headers
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            title_candidates.append({
                "text": header_match.group(2).strip(),
                "priority": 10,
                "distance": table_start_line - i,
            })
            break
        
        # Check for numbered sections
        numbered_match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)$', line)
        if numbered_match:
            title_candidates.append({
                "text": line,
                "priority": 9,
                "distance": table_start_line - i,
            })
            break
        
        # Check for all-caps titles
        if line.isupper() and len(line.split()) >= 2:
            title_candidates.append({
                "text": line,
                "priority": 8,
                "distance": table_start_line - i,
            })
            break
        
        # Check for "Table X:" patterns
        table_label_match = re.match(r'^Table\s+\d+[:.]\s*(.*)$', line, re.IGNORECASE)
        if table_label_match:
            title_candidates.append({
                "text": line,
                "priority": 10,
                "distance": table_start_line - i,
            })
            break
        
        # Any other non-empty line
        if len(line) > 3 and not line.startswith("|"):
            title_candidates.append({
                "text": line,
                "priority": 5,
                "distance": table_start_line - i,
            })
    
    if title_candidates:
        title_candidates.sort(key=lambda x: (-x["priority"], x["distance"]))
        return title_candidates[0]["text"]
    
    return ""


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


def is_numeric_cell(text: str) -> bool:
    """Check if a cell contains primarily numeric content."""
    if not text or text.strip() == "":
        return False
    
    cleaned = text.replace(",", "").replace("$", "").replace("%", "").strip()
    
    try:
        float(cleaned)
        return True
    except ValueError:
        pass
    
    digit_count = sum(c.isdigit() for c in text)
    total_chars = len(text.replace(" ", ""))
    
    if total_chars > 0 and digit_count / total_chars > 0.5:
        return True
    
    return False


def contains_units(text: str) -> bool:
    """Check if text contains unit indicators."""
    unit_patterns = [
        r'\([^)]*(?:mg|kg|mL|g|L|m|cm|mm|°C|°F|%|ppm|ppb|mol|M)\)',
        r'\b(?:mg|kg|mL|g|L|m|cm|mm)(?:/(?:mg|kg|mL|g|L|m|cm|mm))?\b',
    ]
    
    for pattern in unit_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def detect_table_structure(table_data: Dict[str, Any]) -> str:
    """
    Detect if table is VERTICAL (key-value) or HORIZONTAL (standard).
    
    Returns:
        "vertical" - First column contains keys/roles, subsequent columns contain values
        "horizontal" - Standard table with column headers at top
    """
    headers = table_data["headers"]
    rows = table_data["rows"]
    
    if not rows or len(headers) < 2:
        return "horizontal"
    
    # Check if first column looks like keys/labels
    first_col_values = [row[0] for row in rows if row]
    
    # Heuristics for vertical table:
    # 1. First column has mostly text (not numeric)
    # 2. First column values are unique (like keys)
    # 3. Subsequent columns have more varied content
    # 4. First column values look like labels/roles
    
    numeric_in_first_col = sum(1 for val in first_col_values if is_numeric_cell(val))
    non_empty_first_col = sum(1 for val in first_col_values if val.strip())
    
    if non_empty_first_col == 0:
        return "horizontal"
    
    # If first column is mostly numeric, it's likely horizontal
    if numeric_in_first_col / non_empty_first_col > 0.5:
        return "horizontal"
    
    # Check if first column values are unique (like keys)
    unique_ratio = len(set(first_col_values)) / len(first_col_values) if first_col_values else 0
    
    # Check for label-like patterns in first column
    label_keywords = ['name', 'role', 'type', 'category', 'description', 'title', 
                      'director', 'assessment', 'pathology', 'review', 'scientist']
    has_label_keywords = any(
        any(keyword in val.lower() for keyword in label_keywords)
        for val in first_col_values if val
    )
    
    # Decision logic
    if unique_ratio > 0.7 and has_label_keywords:
        return "vertical"
    
    # Check if headers look like they're actually data
    # (This happens when vertical tables are misidentified)
    header_looks_like_data = any(
        len(h) > 50 or '@' in h or 'Phone:' in h
        for h in headers
    )
    
    if header_looks_like_data:
        return "vertical"
    
    return "horizontal"


def detect_header_rows_horizontal(rows: List[List[str]]) -> int:
    """
    Detect how many rows at the top are header rows for HORIZONTAL tables.
    """
    if not rows:
        return 0
    
    max_header_rows = min(5, len(rows))
    header_row_count = 1
    
    for i in range(max_header_rows):
        row = rows[i]
        
        numeric_count = sum(1 for cell in row if is_numeric_cell(cell))
        non_empty_count = sum(1 for cell in row if cell.strip())
        
        if non_empty_count == 0:
            continue
        
        if numeric_count / non_empty_count > 0.5:
            break
        
        has_units = any(contains_units(cell) for cell in row)
        
        header_keywords = ['group', 'number', 'name', 'id', 'type', 'description', 
                          'date', 'time', 'status', 'dose', 'concentration', 'volume', 'animal']
        has_header_keywords = any(
            any(keyword in cell.lower() for keyword in header_keywords)
            for cell in row
        )
        
        if has_units or has_header_keywords or numeric_count == 0:
            header_row_count = i + 1
        else:
            break
    
    return header_row_count


def extract_vertical_table(
    table_data: Dict[str, Any],
    table_index: int,
    source: str = "markdown",
    title: str = ""
) -> Dict[str, Any]:
    """
    Extract VERTICAL key-value table where first column contains keys.
    
    Structure:
    | Key1 | Value1 |
    | Key2 | Value2 |
    | Key3 | Value3 |
    """
    headers = table_data["headers"]
    rows = table_data["rows"]
    
    # In vertical tables, the "headers" row is actually the first data row
    all_data_rows = [headers] + rows
    
    num_columns = len(headers)
    num_rows = len(all_data_rows)
    
    # Create simple column headers: "Column 0", "Column 1", etc.
    # Or use generic names like "Key", "Value"
    if num_columns == 2:
        column_names = ["Key", "Value"]
    else:
        column_names = [f"Column {i}" for i in range(num_columns)]
    
    # Build cells
    cells = []
    for row_idx, row in enumerate(all_data_rows):
        for col_idx, cell_text in enumerate(row):
            cells.append({
                "row": row_idx,
                "col": col_idx,
                "header": column_names[col_idx] if col_idx < len(column_names) else f"Column {col_idx}",
                "header_levels": [column_names[col_idx]] if col_idx < len(column_names) else [f"Column {col_idx}"],
                "text": clean_cell_text(cell_text),
                "is_numeric": is_numeric_cell(cell_text),
            })
    
    return {
        "index": table_index,
        "source": source,
        "title": title,
        "table_type": "vertical",
        "num_columns": num_columns,
        "num_rows": num_rows,
        "header_row_count": 0,  # No header rows in vertical tables
        "headers": column_names,
        "column_headers": [
            {
                "col": i,
                "levels": [name],
                "full_path": name,
            }
            for i, name in enumerate(column_names)
        ],
        "column_groups": [],
        "header_cells": [],
        "cells": cells,
    }


def build_column_header_hierarchy(
    rows: List[List[str]], 
    header_row_count: int
) -> List[Dict[str, Any]]:
    """Build hierarchical header structure for each column."""
    if header_row_count == 0 or not rows:
        return []
    
    num_columns = len(rows[0]) if rows else 0
    column_headers = []
    
    for col_idx in range(num_columns):
        levels = []
        
        for row_idx in range(header_row_count):
            if row_idx < len(rows) and col_idx < len(rows[row_idx]):
                cell_text = clean_cell_text(rows[row_idx][col_idx])
                levels.append(cell_text)
            else:
                levels.append("")
        
        # Remove empty levels from the end
        while levels and not levels[-1]:
            levels.pop()
        
        if not levels:
            levels = [f"Column_{col_idx}"]
        
        non_empty_levels = [l for l in levels if l]
        full_path = " | ".join(non_empty_levels) if non_empty_levels else f"Column_{col_idx}"
        
        column_headers.append({
            "col": col_idx,
            "levels": levels,
            "full_path": full_path,
        })
    
    return column_headers


def detect_grouped_columns(column_headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect column groups based on shared parent headers."""
    if not column_headers:
        return []
    
    groups = []
    max_levels = max(len(ch["levels"]) for ch in column_headers)
    
    if max_levels <= 1:
        return []
    
    current_group = None
    
    for ch in column_headers:
        if not ch["levels"]:
            continue
        
        parent = ch["levels"][0] if ch["levels"] else ""
        
        if current_group is None or current_group["parent"] != parent:
            if current_group is not None:
                groups.append(current_group)
            
            current_group = {
                "parent": parent,
                "start_col": ch["col"],
                "end_col": ch["col"],
                "columns": [ch["col"]],
            }
        else:
            current_group["end_col"] = ch["col"]
            current_group["columns"].append(ch["col"])
    
    if current_group is not None:
        groups.append(current_group)
    
    groups = [g for g in groups if len(g["columns"]) > 1]
    
    return groups


def extract_horizontal_table(
    table_data: Dict[str, Any],
    table_index: int,
    source: str = "markdown",
    title: str = ""
) -> Dict[str, Any]:
    """
    Extract HORIZONTAL standard table with headers at top.
    """
    all_rows = [table_data["headers"]] + table_data["rows"]
    
    header_row_count = detect_header_rows_horizontal(all_rows)
    column_headers = build_column_header_hierarchy(all_rows, header_row_count)
    column_groups = detect_grouped_columns(column_headers)
    
    data_rows = all_rows[header_row_count:]
    
    cells = []
    for row_idx, row in enumerate(data_rows):
        for col_idx, cell_text in enumerate(row):
            if col_idx >= len(column_headers):
                continue
            
            col_header = column_headers[col_idx]
            
            cells.append({
                "row": row_idx,
                "col": col_idx,
                "header": col_header["full_path"],
                "header_levels": col_header["levels"],
                "text": clean_cell_text(cell_text),
                "is_numeric": is_numeric_cell(cell_text),
            })
    
    header_cells = []
    for row_idx in range(header_row_count):
        if row_idx >= len(all_rows):
            break
        for col_idx, cell_text in enumerate(all_rows[row_idx]):
            if col_idx >= len(column_headers):
                continue
            
            header_cells.append({
                "header_row": row_idx,
                "col": col_idx,
                "text": clean_cell_text(cell_text),
                "is_header": True,
            })
    
    simple_headers = [ch["full_path"] for ch in column_headers]
    
    return {
        "index": table_index,
        "source": source,
        "title": title,
        "table_type": "horizontal",
        "num_columns": len(column_headers),
        "num_rows": len(data_rows),
        "header_row_count": header_row_count,
        "headers": simple_headers,
        "column_headers": column_headers,
        "column_groups": column_groups,
        "header_cells": header_cells,
        "cells": cells,
    }


def extract_table_auto(
    table_data: Dict[str, Any],
    table_index: int,
    source: str = "markdown",
    title: str = ""
) -> Dict[str, Any]:
    """
    Automatically detect table structure and extract accordingly.
    """
    table_structure = detect_table_structure(table_data)
    
    if table_structure == "vertical":
        return extract_vertical_table(table_data, table_index, source, title)
    else:
        return extract_horizontal_table(table_data, table_index, source, title)


def extract_all_tables_from_pdf(pdf_path: str, output_dir: str = "."):
    """
    Extract ALL tables from PDF with automatic structure detection.
    """
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    
    rendered = converter(pdf_path)
    markdown = rendered.markdown
    
    # Save images if they exist
    if hasattr(rendered, 'images') and rendered.images:
        for img_name, img_obj in rendered.images.items():
            img_path = Path(output_dir) / img_name
            if isinstance(img_obj, Image.Image):
                img_obj.save(img_path)
                print(f"✓ Saved figure: {img_path}")
            else:
                with open(img_path, 'wb') as f:
                    f.write(img_obj)
                print(f"✓ Saved figure: {img_path}")
    
    # Extract markdown tables
    markdown_tables = parse_markdown_tables_with_context(markdown)
    print(f"\nFound {len(markdown_tables)} markdown tables")
    
    all_tables = []
    
    # Process markdown tables with automatic structure detection
    for t_idx, t in enumerate(markdown_tables):
        title = t.get("title", "")
        table_info = extract_table_auto(
            t,
            table_index=len(all_tables),
            source="markdown",
            title=title
        )
        all_tables.append(table_info)
        
        structure_type = table_info.get("table_type", "unknown")
        print(f"\n✓ Table {table_info['index']} ({structure_type})")
        if title:
            print(f"  Title: {title}")
        print(f"  Size: {table_info['num_rows']} rows × {table_info['num_columns']} cols")
    
    return all_tables, markdown


def print_table_summary(table: Dict[str, Any]):
    """Print detailed summary of a table structure."""
    print(f"\n{'=' * 80}")
    print(f"TABLE {table['index']} SUMMARY")
    print(f"Type: {table.get('table_type', 'unknown').upper()}")
    print(f"Source: {table['source']}")
    print("=" * 80)
    
    if table.get("title"):
        print(f"Title: {table['title']}")
    
    print(f"Data rows: {table['num_rows']}")
    print(f"Columns: {table['num_columns']}")
    print(f"Header rows: {table['header_row_count']}")
    
    if table.get("column_groups"):
        print(f"\nColumn Groups: {len(table['column_groups'])}")
        for group in table["column_groups"]:
            print(f"  - '{group['parent']}': columns {group['start_col']}-{group['end_col']}")
    
    print(f"\nColumn Headers:")
    for ch in table.get("column_headers", []):
        levels_str = " → ".join([f"'{l}'" for l in ch["levels"] if l])
        print(f"  Col {ch['col']}: {levels_str}")
    
    print(f"\nFirst 5 data cells:")
    for cell in table["cells"][:5]:
        print(f"  [{cell['row']}, {cell['col']}] {cell['header']}: '{cell['text']}'")


if __name__ == "__main__":
    pdf_path = "dont_sync/Study Reports/DN23114/Input/Source Files/dn23114-protocol-amend04.pdf"
    output_dir = "./results"
    
    print("Extracting tables with automatic structure detection...\n")
    
    tables, full_markdown = extract_all_tables_from_pdf(pdf_path, output_dir)
    
    for table in tables:
        print_table_summary(table)
    
    # Save to JSON
    output_file = f'{output_dir}/all_tables_output.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved detailed table data to {output_file}")