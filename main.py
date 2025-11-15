
import logging
import time
from pathlib import Path
import pandas as pd

from docling.document_converter import DocumentConverter

from marker.converters.table import TableConverter
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import json
import re
from typing import List, Dict, Any

import camelot
print(f"Using camelot v{camelot.__version__}.")
kwargs = {}

_log = logging.getLogger(__name__)

def parse_markdown_tables(markdown: str) -> List[Dict[str, Any]]:
    """
    Parse all GitHub-style markdown tables from a markdown string.
    Returns a list of tables, each with 'headers' and 'rows'.
    """
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
                tables.append(table)

                # Skip past this table
                i = j
                continue

        i += 1

    return tables

def _split_markdown_row(row: str) -> List[str]:
    """
    Split a markdown table row into cells.
    Example: '| a | b |' -> ['a', 'b']
    """
    # remove leading/trailing pipe
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    cells = [c.strip() for c in row.split("|")]
    return cells

def _parse_single_markdown_table(lines: List[str]) -> Dict[str, Any]:
    """
    Parse a single markdown table (list of lines) into headers/rows.
    lines[0] -> header row
    lines[1] -> separator row
    lines[2:] -> data rows
    """
    header_cells = _split_markdown_row(lines[0])
    # we ignore the separator line (lines[1])

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

def extract_tables_from_pdf_via_markdown(pdf_path: str):
    """
    Use marker-pdf to produce markdown, then parse markdown tables.
    Returns a list of tables with cell-level info.
    """
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)

    rendered = converter(pdf_path)  # MarkdownOutput
    markdown = rendered.markdown

    raw_tables = parse_markdown_tables(markdown)

    # Convert to a more explicit cell-level structure
    structured_tables = []
    for t_idx, t in enumerate(raw_tables):
        table_info = {
            "index": t_idx,
            "headers": [],
            "cells": [],  # list of dicts: row, col, text
        }

        # Clean header cells
        headers = [clean_cell_text(h) for h in t["headers"]]
        table_info["headers"] = headers

        # Data rows
        for r_idx, row in enumerate(t["rows"]):
            for c_idx, cell in enumerate(row):
                cell_text = clean_cell_text(cell)
                table_info["cells"].append(
                    {
                        "row": r_idx,          # 0-based data row index
                        "col": c_idx,
                        "header": headers[c_idx] if c_idx < len(headers) else None,
                        "text": cell_text,
                    }
                )

        structured_tables.append(table_info)

    return structured_tables

def clean_cell_text(text: str) -> str:
    """
    Clean up cell text: remove <br>, convert to spaces or newlines, strip.
    """
    # You can choose to keep newlines instead of spaces if needed
    text = text.replace("<br>", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tables_to_dataframes_from_markdown(structured_tables):
    dfs = []
    for t in structured_tables:
        headers = t["headers"]
        # Build row-wise dicts
        rows_by_index = {}
        for c in t["cells"]:
            r = c["row"]
            col_idx = c["col"]
            if r not in rows_by_index:
                rows_by_index[r] = {h: "" for h in headers}
            if col_idx < len(headers):
                header = headers[col_idx]
                rows_by_index[r][header] = c["text"]

        # Sort rows by row index
        ordered_rows = [rows_by_index[r] for r in sorted(rows_by_index)]
        df = pd.DataFrame(ordered_rows, columns=headers)
        dfs.append({"index": t["index"], "df": df})
    return dfs

def extract_tables_with_cells(pdf_path: str):
    """
    Extract tables from a PDF using marker-pdf's TableConverter.
    Returns a list of tables with cell-level information.
    """
    model_dict = create_model_dict()
    table_converter = TableConverter(artifact_dict=model_dict)

    # IMPORTANT: pass the PDF path, not the PdfConverter output
    result = table_converter(pdf_path)
    print(type(result))
    print("has tables:", hasattr(result, "tables"))
    if hasattr(result, "tables") and result.tables:
        t0 = result.tables[0]
        print("Table attrs:", [a for a in dir(t0) if not a.startswith("_")])

    tables_data = []

    # `result.tables` should be a list of table objects
    for t_idx, table in enumerate(getattr(result, "tables", [])):
        table_info = {
            "index": t_idx,
            "page": getattr(table, "page_num", None),
            "bbox": getattr(table, "bbox", None),
            "cells": [],
        }

        # Try both `.cells` and `.table_cells` depending on version
        cells = getattr(table, "cells", None) or getattr(table, "table_cells", None)
        if not cells:
            continue

        for cell in cells:
            row = getattr(cell, "row", getattr(cell, "row_id", None))
            col = getattr(cell, "col", getattr(cell, "col_id", None))

            cell_data = {
                "row": int(row) if row is not None else None,
                "col": int(col) if col is not None else None,
                "rowspan": getattr(cell, "rowspan", 1),
                "colspan": getattr(cell, "colspan", 1),
                "text": (getattr(cell, "text", "") or "").strip(),
                "bbox": getattr(cell, "bbox", None),
            }
            table_info["cells"].append(cell_data)

        # Optionally keep the markdown representation if available
        if hasattr(table, "markdown"):
            table_info["markdown"] = table.markdown

        tables_data.append(table_info)

    return tables_data

def extract_tables_as_dataframes(pdf_path: str):
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    rendered = converter(pdf_path)

    dataframes = []

    for t_idx, table in enumerate(getattr(rendered, "tables", [])):
        cells = getattr(table, "cells", None) or getattr(table, "table_cells", None)
        if not cells:
            continue

        # infer max row / col (0- or 1-based depending on marker version)
        rows = [
            getattr(c, "row", getattr(c, "row_id", None))
            for c in cells
        ]
        cols = [
            getattr(c, "col", getattr(c, "col_id", None))
            for c in cells
        ]

        # drop None and cast to int
        rows = [int(r) for r in rows if r is not None]
        cols = [int(c) for c in cols if c is not None]
        if not rows or not cols:
            continue

        max_row = max(rows)
        max_col = max(cols)

        # assume rows/cols are 0-based. If you see off-by-one, change to +1 here
        matrix = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

        for cell in cells:
            r = getattr(cell, "row", getattr(cell, "row_id", None))
            c = getattr(cell, "col", getattr(cell, "col_id", None))
            if r is None or c is None:
                continue
            text = (getattr(cell, "text", "") or "").strip()
            matrix[int(r)][int(c)] = text

        # Optionally treat first row as header
        df = pd.DataFrame(matrix[1:], columns=matrix[0])

        dataframes.append(
            {
                "index": t_idx,
                "page": getattr(table, "page_num", None),
                "dataframe": df,
                "bbox": getattr(table, "bbox", None),
            }
        )

    return dataframes

def display_parse_results(tables, parse_time, flavor):
    if not tables:
        return
    tables_dims = ", ".join(
        map(
            lambda table: "{rows}x{cols}".format(
                rows=table.shape[0],
                cols=table.shape[1],
            ),
            tables,
        )
    )
    print(
        f"The {flavor} parser found {len(tables)} table(s) ({tables_dims}) in {parse_time:.2f}s"
    )

    for table in tables:
        print(table.df)

def use_docling(input_doc_path: Path, output_dir: Path):
    doc_converter = DocumentConverter()
    conv_res = doc_converter.convert(input_doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    doc_filename = conv_res.input.file.stem

    # Export tables
    for table_ix, table in enumerate(conv_res.document.tables):
        table_df: pd.DataFrame = table.export_to_dataframe()
        print(f"## Table {table_ix}")
        print(table_df.to_markdown())

        # Save the table as CSV
        element_csv_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.csv"
        _log.info(f"Saving CSV table to {element_csv_filename}")
        table_df.to_csv(element_csv_filename)

        # Save the table as HTML
        element_html_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.html"
        _log.info(f"Saving HTML table to {element_html_filename}")
        with element_html_filename.open("w") as fp:
            fp.write(table.export_to_html(doc=conv_res.document))

def use_marker_simple(input_doc_path: Path, output_dir: Path):
    # 1. Initialize the converter with a model dictionary
    converter = TableConverter(
        artifact_dict=create_model_dict(),
    )

    # 2. Run the converter on your PDF file
    # Replace "FILEPATH" with the actual path to your PDF document
    rendered = converter(input_doc_path.resolve().as_posix())

    # 3. Extract the text (in markdown format) from the rendered output
    text, _, images = text_from_rendered(rendered)

    # The 'text' variable now contains the extracted tables in markdown format
    print(text)

def use_marker(input_doc_path: Path, output_dir: Path):
    # Method 1: Get detailed cell information
    pdf_path = input_doc_path.resolve().as_posix()
    tables = extract_tables_from_pdf_via_markdown(pdf_path)
    
    print(f"Found {len(tables)} tables")
    for t in tables:
        print(f"\nTable {t['index']} with {len(t['cells'])} cells")
        print("Headers:", t["headers"])
        for c in t["cells"]:
            print(f"  row={c['row']} col={c['col']} header={c['header']!r} text={c['text']!r}")
    
    # Save to JSON
    with open('tables_output.json', 'w', encoding='utf-8') as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    
    # Method 2: Get as DataFrames
    dfs = tables_to_dataframes_from_markdown(tables)
    
    for t in dfs:
        print(f"\nTable {t['index']}")
        print(t["df"])
        t["df"].to_csv(f"table_{t['index']}.csv", index=False)

def use_camelot(input_doc_path: Path, output_dir: Path):
    pdf_path = input_doc_path.resolve().as_posix()
    flavor = "hybrid"
    timer_before_parse = time.perf_counter()
    tables = camelot.read_pdf(
        pdf_path,
        flavor=flavor,
        **kwargs,
    )
    timer_after_parse = time.perf_counter()

    display_parse_results(tables, timer_after_parse - timer_before_parse, flavor)

def main():
    logging.basicConfig(level=logging.INFO)
    
    data_folder = Path(__file__).parent / "input"
    input_doc_path = data_folder / "sample-4.pdf"
    output_dir = Path("scratch")

    start_time = time.time()
    use_marker_simple(input_doc_path=input_doc_path, output_dir=output_dir)
    end_time = time.time()
    _log.info(f"Document converted and tables exported in {end_time:.2f} seconds.")

if __name__ == "__main__":
    main()