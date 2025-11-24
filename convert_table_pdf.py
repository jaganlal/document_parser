import re
import sys
import os
import argparse
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def create_pdf_table(markdown_input, output_filename='output.pdf'):
    # Parse the markdown (Reuse logic)
    lines = [line.strip() for line in markdown_input.strip().split('\n')]
    lines = [line for line in lines if line.strip()]
    
    if not lines:
        print("No content found in markdown input.")
        return

    separator_index = -1
    for i, line in enumerate(lines):
        if re.match(r'^\|?[\s\-:|]+\|?$', line) and set(line) & set('-:'):
            separator_index = i
            break
    
    header_rows = []
    data_rows = []
    
    if separator_index != -1:
        header_rows = [line for line in lines[:separator_index]]
        data_rows = [line for line in lines[separator_index+1:] if line.strip().startswith('|')]
    else:
        header_rows = [lines[0]]
        data_rows = lines[1:]

    def parse_row(line):
        content = line.strip()
        if content.startswith('|'): content = content[1:]
        if content.endswith('|'): content = content[:-1]
        return [cell.strip() for cell in content.split('|')]

    parsed_header_rows = [parse_row(row) for row in header_rows]
    parsed_data_rows = [parse_row(row) for row in data_rows]
    
    all_rows = parsed_header_rows + parsed_data_rows
    if not all_rows:
        return
    
    num_cols = max(len(row) for row in all_rows)
    # Pad rows
    for row in all_rows:
        while len(row) < num_cols:
            row.append("")
            
    num_rows = len(all_rows)

    # Merge Logic for ReportLab
    # We need to calculate spans: list of (start_col, start_row, end_col, end_row)
    # Coordinate system for ReportLab SPAN is (col, row).
    
    owners = {} # (r, c) -> (owner_r, owner_c)
    
    for r in range(num_rows):
        for c in range(num_cols):
            owners[(r,c)] = (r,c)
            
            text = all_rows[r][c]
            if not text:
                # Try Merge Up
                if r > 0:
                    owners[(r,c)] = owners[(r-1, c)]
                # Else Try Merge Left
                elif c > 0:
                    owners[(r,c)] = owners[(r, c-1)]
    
    # Group by owner to find extents
    spans = {} # (owner_r, owner_c) -> {min_r, max_r, min_c, max_c}
    
    for r in range(num_rows):
        for c in range(num_cols):
            owner = owners[(r,c)]
            if owner not in spans:
                spans[owner] = {'min_r': r, 'max_r': r, 'min_c': c, 'max_c': c}
            else:
                s = spans[owner]
                s['min_r'] = min(s['min_r'], r)
                s['max_r'] = max(s['max_r'], r)
                s['min_c'] = min(s['min_c'], c)
                s['max_c'] = max(s['max_c'], c)
                
    # Generate Table Data
    # We need to replace <br> with <br/> for Paragraph or newlines?
    # ReportLab Paragraph supports <br/>.
    styles = getSampleStyleSheet()
    styleN = styles['BodyText']
    styleN.alignment = 1 # Center
    
    table_data = []
    for r in range(num_rows):
        row_data = []
        for c in range(num_cols):
            # Only add content if this is the owner cell (top-left)
            # Actually ReportLab Table expects data in all cells, but spanned cells are ignored visually.
            # But we should put the text in the top-left cell of the span.
            
            owner = owners[(r,c)]
            if owner == (r,c):
                text = all_rows[r][c]
                # Replace <br> with <br/>
                text = text.replace('<br>', '<br/>')
                # Wrap in Paragraph for wrapping and formatting
                # Bold headers?
                if r < len(parsed_header_rows):
                    text = f"<b>{text}</b>"
                
                p = Paragraph(text, styleN)
                row_data.append(p)
            else:
                row_data.append("") # Placeholder
        table_data.append(row_data)

    # Build Table Style
    tbl_style = TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ])
    
    # Add Spans
    for owner, s in spans.items():
        if s['max_r'] > s['min_r'] or s['max_c'] > s['min_c']:
            # ReportLab SPAN: (sc, sr), (ec, er)
            sc, sr = s['min_c'], s['min_r']
            ec, er = s['max_c'], s['max_r']
            tbl_style.add('SPAN', (sc, sr), (ec, er))

    # Create PDF
    doc = SimpleDocTemplate(output_filename, pagesize=landscape(letter))
    elements = []
    
    # Title
    elements.append(Paragraph("<b>Converted Table</b>", styles['Heading2']))
    
    t = Table(table_data)
    t.setStyle(tbl_style)
    elements.append(t)
    
    doc.build(elements)
    print(f"Successfully created {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Convert Markdown table to PDF")
    parser.add_argument('input', help="Markdown string or file path")
    parser.add_argument('--output', default='output.pdf', help="Output file name")
    
    args = parser.parse_args()
    
    markdown_input = args.input
    if os.path.isfile(args.input):
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                markdown_input = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
            
    create_pdf_table(markdown_input, args.output)

if __name__ == "__main__":
    main()
