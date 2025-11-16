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
    """
    Extract table title by looking at lines before the table.
    
    Heuristics:
    1. Look at previous 5 lines
    2. Find headers (##, ###) or lines with specific patterns
    3. Skip empty lines
    4. Return the closest non-empty line that looks like a title
    """
    title_candidates = []
    
    # Look backwards up to 5 lines
    lookback = min(5, table_start_line)
    
    for i in range(table_start_line - 1, table_start_line - lookback - 1, -1):
        if i < 0:
            break
        
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip figure references
        if line.startswith("![]"):
            continue
        
        # Skip horizontal rules
        if re.match(r'^[-=*_]{3,}$', line):
            continue
        
        # Check for markdown headers (##, ###, etc.)
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            title_candidates.append({
                "text": header_match.group(2).strip(),
                "priority": 10,  # High priority for markdown headers
                "distance": table_start_line - i,
            })
            break  # Stop at first header
        
        # Check for numbered sections (e.g., "3 CONTRIBUTING SCIENTISTS", "10.4 Experimental Design")
        numbered_match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)$', line)
        if numbered_match:
            title_candidates.append({
                "text": line,
                "priority": 9,  # High priority for numbered sections
                "distance": table_start_line - i,
            })
            break
        
        # Check for all-caps titles (common in documents)
        if line.isupper() and len(line.split()) >= 2:
            title_candidates.append({
                "text": line,
                "priority": 8,
                "distance": table_start_line - i,
            })
            break
        
        # Check for "Table X:" or "Table X." patterns
        table_label_match = re.match(r'^Table\s+\d+[:.]\s*(.*)$', line, re.IGNORECASE)
        if table_label_match:
            title_candidates.append({
                "text": line,
                "priority": 10,
                "distance": table_start_line - i,
            })
            break
        
        # Any other non-empty line (lower priority)
        if len(line) > 3 and not line.startswith("|"):
            title_candidates.append({
                "text": line,
                "priority": 5,
                "distance": table_start_line - i,
            })
    
    # Select best title candidate
    if title_candidates:
        # Sort by priority (desc) then distance (asc)
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


def is_numeric_cell(text: str) -> bool:
    """Check if a cell contains primarily numeric content."""
    if not text or text.strip() == "":
        return False
    
    # Remove common numeric formatting
    cleaned = text.replace(",", "").replace("$", "").replace("%", "").strip()
    
    # Check if it's a number (int or float)
    try:
        float(cleaned)
        return True
    except ValueError:
        pass
    
    # Check if it contains mostly digits
    digit_count = sum(c.isdigit() for c in text)
    total_chars = len(text.replace(" ", ""))
    
    if total_chars > 0 and digit_count / total_chars > 0.5:
        return True
    
    return False


def contains_units(text: str) -> bool:
    """Check if text contains unit indicators (parentheses with units)."""
    unit_patterns = [
        r'\([^)]*(?:mg|kg|mL|g|L|m|cm|mm|°C|°F|%|ppm|ppb|mol|M)\)',
        r'\b(?:mg|kg|mL|g|L|m|cm|mm)(?:/(?:mg|kg|mL|g|L|m|cm|mm))?\b',
    ]
    
    for pattern in unit_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def detect_header_rows(rows: List[List[str]]) -> int:
    """
    Detect how many rows at the top are header rows.
    
    Heuristics:
    1. Rows with mostly non-numeric content
    2. Rows containing units in parentheses
    3. Rows before the first all-numeric row
    4. Maximum of 5 header rows (safety limit)
    """
    if not rows:
        return 0
    
    max_header_rows = min(5, len(rows))
    header_row_count = 1  # At least one header row
    
    for i in range(max_header_rows):
        row = rows[i]
        
        # Count numeric vs non-numeric cells
        numeric_count = sum(1 for cell in row if is_numeric_cell(cell))
        non_empty_count = sum(1 for cell in row if cell.strip())
        
        if non_empty_count == 0:
            continue
        
        # If mostly numeric, this is likely a data row
        if numeric_count / non_empty_count > 0.5:
            break
        
        # Check for units (common in header rows)
        has_units = any(contains_units(cell) for cell in row)
        
        # Check for common header keywords
        header_keywords = ['group', 'number', 'name', 'id', 'type', 'description', 
                          'date', 'time', 'status', 'role', 'details', 'dose', 
                          'concentration', 'volume', 'animal']
        has_header_keywords = any(
            any(keyword in cell.lower() for keyword in header_keywords)
            for cell in row
        )
        
        if has_units or has_header_keywords or numeric_count == 0:
            header_row_count = i + 1
        else:
            break
    
    return header_row_count


def build_column_header_hierarchy(
    rows: List[List[str]], 
    header_row_count: int
) -> List[Dict[str, Any]]:
    """
    Build hierarchical header structure for each column.
    
    Returns:
        List of dicts with 'col', 'levels', and 'full_path' for each column
    """
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
        
        # If all levels are empty, use a placeholder
        if not levels:
            levels = [f"Column_{col_idx}"]
        
        # Create full path (join non-empty levels)
        non_empty_levels = [l for l in levels if l]
        full_path = " | ".join(non_empty_levels) if non_empty_levels else f"Column_{col_idx}"
        
        column_headers.append({
            "col": col_idx,
            "levels": levels,
            "full_path": full_path,
        })
    
    return column_headers


def detect_grouped_columns(column_headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect column groups based on shared parent headers.
    
    Returns:
        List of group dicts with 'parent', 'start_col', 'end_col', 'columns'
    """
    if not column_headers:
        return []
    
    groups = []
    
    # Check if we have multi-level headers
    max_levels = max(len(ch["levels"]) for ch in column_headers)
    
    if max_levels <= 1:
        return []
    
    # Group by parent level (level 0)
    current_group = None
    
    for ch in column_headers:
        if not ch["levels"]:
            continue
        
        parent = ch["levels"][0] if ch["levels"] else ""
        
        if current_group is None or current_group["parent"] != parent:
            # Start new group
            if current_group is not None:
                groups.append(current_group)
            
            current_group = {
                "parent": parent,
                "start_col": ch["col"],
                "end_col": ch["col"],
                "columns": [ch["col"]],
            }
        else:
            # Extend current group
            current_group["end_col"] = ch["col"]
            current_group["columns"].append(ch["col"])
    
    # Add last group
    if current_group is not None:
        groups.append(current_group)
    
    # Filter out single-column "groups"
    groups = [g for g in groups if len(g["columns"]) > 1]
    
    return groups


def extract_table_with_complex_headers(
    table_data: Dict[str, Any],
    table_index: int,
    source: str = "markdown",
    title: str = ""
) -> Dict[str, Any]:
    """
    Extract table with support for multi-row headers and grouped columns.
    
    Args:
        table_data: Dict with 'headers' and 'rows' from markdown parser
        table_index: Index of this table
        source: Source type ('markdown' or 'figure_ocr')
        title: Table title/caption
    
    Returns:
        Enhanced table dict with header hierarchy and title
    """
    all_rows = [table_data["headers"]] + table_data["rows"]
    
    # Detect header rows
    header_row_count = detect_header_rows(all_rows)
    
    # Build column header hierarchy
    column_headers = build_column_header_hierarchy(all_rows, header_row_count)
    
    # Detect grouped columns
    column_groups = detect_grouped_columns(column_headers)
    
    # Extract data rows (after header rows)
    data_rows = all_rows[header_row_count:]
    
    # Build cell-level structure
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
    
    # Build header cells (for reference)
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
    
    num_columns = len(column_headers)
    num_data_rows = len(data_rows)
    
    # Create simple flat headers list for backward compatibility
    simple_headers = [ch["full_path"] for ch in column_headers]
    
    return {
        "index": table_index,
        "source": source,
        "title": title,  # NEW: Table title/caption
        "num_columns": num_columns,
        "num_rows": num_data_rows,
        "header_row_count": header_row_count,
        "headers": simple_headers,  # Backward compatible
        "column_headers": column_headers,  # New: hierarchical headers
        "column_groups": column_groups,  # New: grouped columns
        "header_cells": header_cells,  # New: header cell data
        "cells": cells,  # Enhanced with header_levels
    }


def extract_all_tables_and_figures(pdf_path: str, output_dir: str = "."):
    """
    Extract ALL tables from PDF including those detected as figures.
    Supports multi-row headers, grouped columns, and table titles.
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
    
    print("\n" + "=" * 80)
    print("FULL MARKDOWN OUTPUT:")
    print("=" * 80)
    print(markdown)
    print("=" * 80)
    
    # Extract markdown tables with context (titles)
    markdown_tables = parse_markdown_tables_with_context(markdown)
    print(f"\nFound {len(markdown_tables)} markdown tables")
    
    # Extract figure references
    figure_refs = extract_figure_references(markdown)
    print(f"Found {len(figure_refs)} figures")
    
    all_tables = []
    
    # Process figures (potential tables)
    markdown_lines = markdown.splitlines()
    for fig_idx, fig_ref in enumerate(figure_refs):
        print(f"\nProcessing figure: {fig_ref}")
        fig_path = Path(output_dir) / fig_ref
        
        # Try to find title for this figure
        fig_title = ""
        for line_idx, line in enumerate(markdown_lines):
            if fig_ref in line:
                fig_title = _extract_table_title(markdown_lines, line_idx)
                break
        
        if fig_path.exists():
            ocr_text = ocr_table_from_image(str(fig_path))
            print(f"OCR text from {fig_ref}:")
            print(ocr_text)
            print("-" * 40)
            
            parsed_table = parse_ocr_table_text(ocr_text)
            if parsed_table:
                table_info = extract_table_with_complex_headers(
                    parsed_table,
                    table_index=len(all_tables),
                    source="figure_ocr",
                    title=fig_title
                )
                table_info["figure_path"] = fig_ref
                all_tables.append(table_info)
                print(f"✓ Extracted table from figure with {table_info['num_rows']} data rows")
                if fig_title:
                    print(f"  Title: {fig_title}")
            else:
                print(f"⚠ Could not parse table structure from OCR text")
        else:
            print(f"⚠ Figure file not found: {fig_path}")
    
    # Process markdown tables
    for t_idx, t in enumerate(markdown_tables):
        title = t.get("title", "")
        table_info = extract_table_with_complex_headers(
            t,
            table_index=len(all_tables),
            source="markdown",
            title=title
        )
        all_tables.append(table_info)
        if title:
            print(f"\n✓ Table {table_info['index']} title: {title}")
    
    return all_tables, markdown


def tables_to_dataframes(structured_tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert structured tables to pandas DataFrames."""
    dfs = []
    
    for t in structured_tables:
        # Use hierarchical headers if available
        if t.get("column_headers"):
            # Create MultiIndex columns if we have multi-level headers
            max_levels = max(len(ch["levels"]) for ch in t["column_headers"])
            
            if max_levels > 1:
                # Build MultiIndex
                header_arrays = []
                for level_idx in range(max_levels):
                    level_headers = []
                    for ch in t["column_headers"]:
                        if level_idx < len(ch["levels"]):
                            level_headers.append(ch["levels"][level_idx])
                        else:
                            level_headers.append("")
                    header_arrays.append(level_headers)
                
                columns = pd.MultiIndex.from_arrays(header_arrays)
            else:
                columns = [ch["full_path"] for ch in t["column_headers"]]
        else:
            columns = t["headers"]
        
        # Build data rows
        rows_by_index = {}
        for c in t["cells"]:
            r = c["row"]
            col_idx = c["col"]
            if r not in rows_by_index:
                rows_by_index[r] = [""] * t["num_columns"]
            if col_idx < t["num_columns"]:
                rows_by_index[r][col_idx] = c["text"]
        
        ordered_rows = [rows_by_index[r] for r in sorted(rows_by_index)]
        
        df = pd.DataFrame(ordered_rows, columns=columns)
        
        dfs.append({
            "index": t["index"],
            "source": t.get("source", "unknown"),
            "title": t.get("title", ""),
            "df": df,
            "shape": df.shape,
            "has_multiindex": isinstance(columns, pd.MultiIndex),
        })
    
    return dfs


def print_table_summary(table: Dict[str, Any]):
    """Print detailed summary of a table structure."""
    print(f"\n{'=' * 80}")
    print(f"TABLE {table['index']} SUMMARY (source: {table['source']})")
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
    
    print(f"\nColumn Headers (hierarchical):")
    for ch in table.get("column_headers", []):
        levels_str = " → ".join([f"'{l}'" for l in ch["levels"] if l])
        print(f"  Col {ch['col']}: {levels_str}")
    
    print(f"\nFirst 5 data cells:")
    for cell in table["cells"][:5]:
        print(f"  [{cell['row']}, {cell['col']}] {cell['header']}: '{cell['text']}'")


if __name__ == "__main__":
    pdf_path = "dont_sync/Study Reports/DN23114/Input/Source Files/dn23114-protocol-amend04.pdf"
    output_dir = "./results"
    
    print("Extracting ALL tables with complex header support and titles...\n")
    
    # Extract all tables
    tables, full_markdown = extract_all_tables_and_figures(pdf_path, output_dir)
    
    # Print summaries
    for table in tables:
        print_table_summary(table)
    
    # Save to JSON
    output_file = f'{output_dir}/all_tables_output.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved detailed table data to {output_file}")
    
    # Convert to DataFrames
    dfs = tables_to_dataframes(tables)
    
    # print(f"\n{'=' * 80}")
    # print("DATAFRAMES:")
    # print("=" * 80)
    
    for t in dfs:
        # print(f"\n--- Table {t['index']} (source: {t['source']}, shape: {t['shape']}) ---")
        # if t['title']:
        #     print(f"Title: {t['title']}")
        # if t['has_multiindex']:
        #     print("(MultiIndex columns)")
        # print(t['df'])
        
        # Save to CSV
        csv_filename = f"table_{t['index']}_{t['source']}.csv"
        t['df'].to_csv(f'{output_dir}/{csv_filename}', index=False)
        print(f"✓ Saved to {csv_filename}")
    
    # Save full markdown
    with open(f'{output_dir}/marker_full_output.md', 'w', encoding='utf-8') as f:
        f.write(full_markdown)
    print(f"\n✓ Saved full markdown to {output_dir}/marker_full_output.md")