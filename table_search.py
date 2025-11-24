"""
Improved Table Search Engine
Works with both VERTICAL (key-value) and HORIZONTAL (standard) tables.
Supports fuzzy matching, case-insensitive search, and flexible querying.
"""

import json
import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class SearchMode(Enum):
    """Search modes for different matching strategies."""
    EXACT = "exact"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    FUZZY = "fuzzy"


@dataclass
class SearchResult:
    """Represents a single search result."""
    table_index: int
    table_source: str
    table_title: str
    table_type: str  # "vertical" or "horizontal"
    row: int
    col: int
    header: str
    header_levels: List[str]
    cell_text: str
    match_score: float = 1.0
    match_type: str = "cell"
    matched_header_level: Optional[int] = None
    context: Optional[Dict[str, Any]] = None


class UnifiedTableSearchEngine:
    """
    Unified search engine that handles both vertical and horizontal tables.
    """
    
    def __init__(self, tables_data: List[Dict[str, Any]]):
        self.tables = tables_data
        self._build_index()
    
    def _build_index(self):
        """Build search index for faster lookups."""
        self.cell_index = []
        self.table_index = []
        
        for table in self.tables:
            table_idx = table.get("index", 0)
            table_source = table.get("source", "unknown")
            table_title = table.get("title", "")
            table_type = table.get("table_type", "horizontal")
            column_headers = table.get("column_headers", [])
            
            self.table_index.append({
                "table_index": table_idx,
                "table_source": table_source,
                "table_title": table_title,
                "table_type": table_type,
                "num_rows": table.get("num_rows", 0),
                "num_columns": table.get("num_columns", 0),
                "header_row_count": table.get("header_row_count", 1),
                "column_groups": table.get("column_groups", []),
            })
            
            for cell in table.get("cells", []):
                col_idx = cell.get("col", 0)
                
                col_header_info = None
                for ch in column_headers:
                    if ch["col"] == col_idx:
                        col_header_info = ch
                        break
                
                header_levels = cell.get("header_levels", [])
                if not header_levels and col_header_info:
                    header_levels = col_header_info.get("levels", [])
                
                self.cell_index.append({
                    "table_index": table_idx,
                    "table_source": table_source,
                    "table_title": table_title,
                    "table_type": table_type,
                    "row": cell.get("row"),
                    "col": col_idx,
                    "header": cell.get("header", ""),
                    "header_levels": header_levels,
                    "text": cell.get("text", ""),
                    "is_numeric": cell.get("is_numeric", False),
                })
    
    def _match_text(
        self, query: str, text: str, mode: SearchMode, case_sensitive: bool
    ) -> Dict[str, Any]:
        """Match query against text based on mode."""
        if not case_sensitive:
            query = query.lower()
            text = text.lower()
        
        if mode == SearchMode.EXACT:
            matched = query == text
            score = 1.0 if matched else 0.0
        
        elif mode == SearchMode.CONTAINS:
            matched = query in text
            score = 1.0 if matched else 0.0
        
        elif mode == SearchMode.STARTS_WITH:
            matched = text.startswith(query)
            score = 1.0 if matched else 0.0
        
        elif mode == SearchMode.ENDS_WITH:
            matched = text.endswith(query)
            score = 1.0 if matched else 0.0
        
        elif mode == SearchMode.REGEX:
            try:
                pattern = re.compile(query, 0 if case_sensitive else re.IGNORECASE)
                matched = bool(pattern.search(text))
                score = 1.0 if matched else 0.0
            except re.error:
                matched = False
                score = 0.0
        
        elif mode == SearchMode.FUZZY:
            score = self._fuzzy_match_score(query, text)
            matched = score > 0.6
        
        else:
            matched = False
            score = 0.0
        
        return {"matched": matched, "score": score}
    
    def _fuzzy_match_score(self, query: str, text: str) -> float:
        """Calculate fuzzy match score using Levenshtein distance."""
        if not query or not text:
            return 0.0
        
        len_q, len_t = len(query), len(text)
        
        if len_q > len_t:
            query, text = text, query
            len_q, len_t = len_t, len_q
        
        current_row = range(len_q + 1)
        for i in range(1, len_t + 1):
            previous_row, current_row = current_row, [i] + [0] * len_q
            for j in range(1, len_q + 1):
                add, delete, change = (
                    previous_row[j] + 1,
                    current_row[j - 1] + 1,
                    previous_row[j - 1],
                )
                if text[i - 1] != query[j - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)
        
        distance = current_row[len_q]
        max_len = max(len(query), len(text))
        similarity = 1 - (distance / max_len)
        
        return similarity
    
    def search_by_key_value(
        self,
        key_query: str,
        table_title: Optional[str] = None,
        mode: SearchMode = SearchMode.CONTAINS,
        case_sensitive: bool = False
    ) -> List[SearchResult]:
        """
        Search for a key in VERTICAL tables and return the corresponding value.
        
        For vertical tables like:
        | Pathology | Dr. Smith |
        | Clinical  | Dr. Jones |
        
        Args:
            key_query: The key to search for (e.g., "Pathology")
            table_title: Optional table title filter
            mode: Search mode
            case_sensitive: Whether search is case-sensitive
        
        Returns:
            List of SearchResult objects with matching key-value pairs
        """
        results = []
        
        for cell_data in self.cell_index:
            # Only search in vertical tables
            if cell_data["table_type"] != "vertical":
                continue
            
            # Filter by table title
            if table_title and table_title.lower() not in cell_data["table_title"].lower():
                continue
            
            # Only search in the first column (keys)
            if cell_data["col"] != 0:
                continue
            
            # Match the key
            match_result = self._match_text(
                key_query, cell_data["text"], mode, case_sensitive
            )
            
            if match_result["matched"]:
                # Find the corresponding value(s) in the same row
                row_num = cell_data["row"]
                table_idx = cell_data["table_index"]
                
                # Get all cells in this row
                row_cells = [
                    c for c in self.cell_index
                    if c["table_index"] == table_idx and c["row"] == row_num
                ]
                
                # Sort by column
                row_cells.sort(key=lambda x: x["col"])
                
                # Create results for each value column
                for cell in row_cells:
                    if cell["col"] > 0:  # Skip the key column
                        results.append(SearchResult(
                            table_index=table_idx,
                            table_source=cell_data["table_source"],
                            table_title=cell_data["table_title"],
                            table_type="vertical",
                            row=row_num,
                            col=cell["col"],
                            header=cell["header"],
                            header_levels=cell["header_levels"],
                            cell_text=cell["text"],
                            match_score=match_result["score"],
                            match_type="key_value",
                            context={
                                "key": cell_data["text"],
                                "key_column": cell_data["header"],
                                "value_column": cell["header"],
                            },
                        ))
        
        return results
    
    def search_by_column(
        self,
        column_name: str,
        value_query: Optional[str] = None,
        table_title: Optional[str] = None,
        mode: SearchMode = SearchMode.CONTAINS,
        case_sensitive: bool = False
    ) -> List[SearchResult]:
        """
        Search for values in a specific column of HORIZONTAL tables.
        
        Args:
            column_name: Name of the column to search in
            value_query: Optional value to search for (if None, returns all values in column)
            table_title: Optional table title filter
            mode: Search mode
            case_sensitive: Whether search is case-sensitive
        
        Returns:
            List of SearchResult objects
        """
        results = []
        
        for cell_data in self.cell_index:
            # Only search in horizontal tables
            if cell_data["table_type"] != "horizontal":
                continue
            
            # Filter by table title
            if table_title and table_title.lower() not in cell_data["table_title"].lower():
                continue
            
            # Check if column matches
            column_match = False
            if cell_data["header"] == column_name:
                column_match = True
            elif column_name in cell_data["header_levels"]:
                column_match = True
            elif any(column_name.lower() in level.lower() for level in cell_data["header_levels"]):
                column_match = True
            
            if not column_match:
                continue
            
            # If value_query is specified, match it
            if value_query is not None:
                match_result = self._match_text(
                    value_query, cell_data["text"], mode, case_sensitive
                )
                if not match_result["matched"]:
                    continue
                score = match_result["score"]
            else:
                score = 1.0
            
            results.append(SearchResult(
                table_index=cell_data["table_index"],
                table_source=cell_data["table_source"],
                table_title=cell_data["table_title"],
                table_type="horizontal",
                row=cell_data["row"],
                col=cell_data["col"],
                header=cell_data["header"],
                header_levels=cell_data["header_levels"],
                cell_text=cell_data["text"],
                match_score=score,
                match_type="column_search",
            ))
        
        return results
    
    def get_row_by_column_value(
        self,
        column_name: str,
        value_query: str,
        table_title: Optional[str] = None,
        mode: SearchMode = SearchMode.CONTAINS,
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Find rows where a column contains a specific value, return entire row.
        
        Example:
            Find rows where "Group Number" = "2", return all columns
        
        Args:
            column_name: Column to search in
            value_query: Value to search for
            table_title: Optional table title filter
            mode: Search mode
            case_sensitive: Whether search is case-sensitive
        
        Returns:
            List of dicts with full row data
        """
        # First, find matching cells
        matching_cells = self.search_by_column(
            column_name, value_query, table_title, mode, case_sensitive
        )
        
        results = []
        seen_rows = set()
        
        for cell in matching_cells:
            row_key = (cell.table_index, cell.row)
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            
            # Get all cells in this row
            row_cells = [
                c for c in self.cell_index
                if c["table_index"] == cell.table_index and c["row"] == cell.row
            ]
            
            # Sort by column
            row_cells.sort(key=lambda x: x["col"])
            
            # Build row data dict
            row_data = {}
            for rc in row_cells:
                row_data[rc["header"]] = rc["text"]
            
            results.append({
                "table_index": cell.table_index,
                "table_title": cell.table_title,
                "table_type": cell.table_type,
                "row": cell.row,
                "matched_column": cell.header,
                "matched_value": cell.cell_text,
                "match_score": cell.match_score,
                "row_data": row_data,
            })
        
        return results
    
    def search_anywhere(
        self,
        query: str,
        table_title: Optional[str] = None,
        table_type: Optional[str] = None,
        mode: SearchMode = SearchMode.CONTAINS,
        case_sensitive: bool = False,
        max_results: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Search for a query anywhere in tables (cells, headers, titles).
        
        Args:
            query: Search query
            table_title: Optional table title filter
            table_type: Optional table type filter ("vertical" or "horizontal")
            mode: Search mode
            case_sensitive: Whether search is case-sensitive
            max_results: Maximum number of results
        
        Returns:
            List of SearchResult objects
        """
        results = []
        
        # Search in cells
        for cell_data in self.cell_index:
            # Apply filters
            if table_title and table_title.lower() not in cell_data["table_title"].lower():
                continue
            
            if table_type and cell_data["table_type"] != table_type:
                continue
            
            # Search in cell text
            match_result = self._match_text(
                query, cell_data["text"], mode, case_sensitive
            )
            
            if match_result["matched"]:
                results.append(SearchResult(
                    table_index=cell_data["table_index"],
                    table_source=cell_data["table_source"],
                    table_title=cell_data["table_title"],
                    table_type=cell_data["table_type"],
                    row=cell_data["row"],
                    col=cell_data["col"],
                    header=cell_data["header"],
                    header_levels=cell_data["header_levels"],
                    cell_text=cell_data["text"],
                    match_score=match_result["score"],
                    match_type="cell",
                ))
        
        # Sort by match score
        results.sort(key=lambda x: (-x.match_score, x.table_index, x.row, x.col))
        
        if max_results:
            results = results[:max_results]
        
        return results
    
    def list_all_tables(self) -> List[Dict[str, Any]]:
        """List all tables with their metadata."""
        return [
            {
                "index": t["table_index"],
                "title": t["table_title"],
                "type": t["table_type"],
                "source": t["table_source"],
                "size": f"{t['num_rows']}x{t['num_columns']}",
            }
            for t in self.table_index
        ]
    
    def get_table_by_title(
        self,
        title_query: str,
        mode: SearchMode = SearchMode.CONTAINS
    ) -> Optional[Dict[str, Any]]:
        """Get the first table matching the title query."""
        for table in self.tables:
            title = table.get("title", "")
            match_result = self._match_text(title_query, title, mode, False)
            if match_result["matched"]:
                return table
        return None


def print_search_results(results: List[SearchResult], max_display: int = 20):
    """Pretty print search results."""
    print(f"\n{'=' * 80}")
    print(f"Found {len(results)} results")
    print("=" * 80)
    
    for i, result in enumerate(results[:max_display]):
        print(f"\n[{i+1}] Table {result.table_index} ({result.table_type.upper()})")
        if result.table_title:
            print(f"    Title: {result.table_title}")
        print(f"    Position: Row {result.row}, Col {result.col}")
        print(f"    Column: {result.header.replace(chr(10), ' ')}")
        print(f"    Value: {result.cell_text.replace(chr(10), ' ')}")
        print(f"    Match Type: {result.match_type}")
        print(f"    Match Score: {result.match_score:.2f}")
        
        if result.context:
            print(f"    Context: {result.context}")
    
    if len(results) > max_display:
        print(f"\n... and {len(results) - max_display} more results")


def print_row_results(results: List[Dict[str, Any]], max_display: int = 20):
    """Pretty print row search results."""
    print(f"\n{'=' * 80}")
    print(f"Found {len(results)} matching rows")
    print("=" * 80)
    
    for i, result in enumerate(results[:max_display]):
        print(f"\n[{i+1}] Table {result['table_index']} ({result['table_type'].upper()})")
        if result.get('table_title'):
            print(f"    Title: {result['table_title']}")
        print(f"    Row: {result['row']}")
        print(f"    Matched: {result['matched_column']} = '{result['matched_value']}'")
        print(f"    Match Score: {result['match_score']:.2f}")
        print(f"\n    Full Row Data:")
        print(f"\n    Full Row Data:")
        for header, value in result['row_data'].items():
            # Handle multi-line values for better display
            header_str = header.replace('\n', ' ')
            value_str = str(value).replace('\n', '\n' + ' ' * (len(header_str) + 8))
            print(f"      {header_str}: {value_str}")
    
    if len(results) > max_display:
        print(f"\n... and {len(results) - max_display} more results")


# Example usage
if __name__ == "__main__":
    # Load table data
    output_dir = "./results"
    with open(f'{output_dir}/all_tables_output.json', 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    
    # Initialize search engine
    engine = UnifiedTableSearchEngine(tables_data)
    
    print("=" * 80)
    print("UNIFIED TABLE SEARCH ENGINE")
    print("Handles both VERTICAL and HORIZONTAL tables automatically")
    print("=" * 80)
    
    # Example 1: List all tables
    print("\n1. List all tables:")
    all_tables = engine.list_all_tables()
    for t in all_tables:
        print(f"  Table {t['index']}: '{t['title']}' ({t['type']}, {t['size']})")
    
    # Example 2: Search for "Pathology" in vertical table
    print("\n2. Search for 'Pathology' key in vertical tables:")
    results = engine.search_by_key_value(
        key_query="Pathology",
        table_title="CONTRIBUTING SCIENTISTS"
    )
    print_search_results(results)
    
    # Example 3: Search by column in horizontal table
    print("\n3. Search for group number '2' in Experimental Design table:")
    results = engine.get_row_by_column_value(
        column_name="Group Number",
        value_query="2",
        table_title="Experimental Design"
    )
    print_row_results(results)
    
    # Example 4: Search anywhere
    print("\n4. Search for 'mg/kg' anywhere:")
    results = engine.search_anywhere(
        query="mg/kg",
        max_results=10
    )
    print_search_results(results)
    
    print("\n" + "=" * 80)
    print("SEARCH ENGINE READY")
    print("=" * 80)
