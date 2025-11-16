import json
import re
from typing import List, Dict, Any, Optional
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
    row: int
    col: int
    header: str
    cell_text: str
    match_score: float = 1.0
    context: Optional[Dict[str, Any]] = None


class TableSearchEngine:
    """Search engine for table data with cell-level granularity."""
    
    def __init__(self, tables_data: List[Dict[str, Any]]):
        """
        Initialize search engine with table data.
        
        Args:
            tables_data: List of table dictionaries with 'cells' structure
        """
        self.tables = tables_data
        self._build_index()
    
    def _build_index(self):
        """Build search index for faster lookups."""
        self.cell_index = []
        
        for table in self.tables:
            table_idx = table.get("index", 0)
            table_source = table.get("source", "unknown")
            
            for cell in table.get("cells", []):
                self.cell_index.append({
                    "table_index": table_idx,
                    "table_source": table_source,
                    "row": cell.get("row"),
                    "col": cell.get("col"),
                    "header": cell.get("header", ""),
                    "text": cell.get("text", ""),
                    "table_headers": table.get("headers", []),
                    "table_num_rows": table.get("num_rows", 0),
                    "table_num_cols": table.get("num_columns", 0),
                })
    
    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.CONTAINS,
        case_sensitive: bool = False,
        search_headers: bool = True,
        search_cells: bool = True,
        table_index: Optional[int] = None,
        column: Optional[str] = None,
        row: Optional[int] = None,
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search tables at cell level.
        
        Args:
            query: Search query string
            mode: Search mode (exact, contains, starts_with, ends_with, regex, fuzzy)
            case_sensitive: Whether search is case-sensitive
            search_headers: Include header names in search
            search_cells: Include cell values in search
            table_index: Filter by specific table index
            column: Filter by column header name
            row: Filter by specific row number
            max_results: Maximum number of results to return
        
        Returns:
            List of SearchResult objects
        """
        results = []
        
        for cell_data in self.cell_index:
            # Apply filters
            if table_index is not None and cell_data["table_index"] != table_index:
                continue
            
            if column is not None and cell_data["header"] != column:
                continue
            
            if row is not None and cell_data["row"] != row:
                continue
            
            # Search in cell text
            if search_cells:
                match_result = self._match_text(
                    query, cell_data["text"], mode, case_sensitive
                )
                if match_result["matched"]:
                    results.append(SearchResult(
                        table_index=cell_data["table_index"],
                        table_source=cell_data["table_source"],
                        row=cell_data["row"],
                        col=cell_data["col"],
                        header=cell_data["header"],
                        cell_text=cell_data["text"],
                        match_score=match_result["score"],
                        context=self._get_cell_context(cell_data),
                    ))
            
            # Search in header names
            if search_headers and cell_data["header"]:
                match_result = self._match_text(
                    query, cell_data["header"], mode, case_sensitive
                )
                if match_result["matched"]:
                    results.append(SearchResult(
                        table_index=cell_data["table_index"],
                        table_source=cell_data["table_source"],
                        row=cell_data["row"],
                        col=cell_data["col"],
                        header=cell_data["header"],
                        cell_text=cell_data["text"],
                        match_score=match_result["score"],
                        context=self._get_cell_context(cell_data),
                    ))
        
        # Sort by match score (descending)
        results.sort(key=lambda x: x.match_score, reverse=True)
        
        # Apply max_results limit
        if max_results:
            results = results[:max_results]
        
        return results
    
    def _match_text(
        self, query: str, text: str, mode: SearchMode, case_sensitive: bool
    ) -> Dict[str, Any]:
        """
        Match query against text based on mode.
        
        Returns:
            Dict with 'matched' (bool) and 'score' (float)
        """
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
            matched = score > 0.6  # Threshold for fuzzy matching
        
        else:
            matched = False
            score = 0.0
        
        return {"matched": matched, "score": score}
    
    def _fuzzy_match_score(self, query: str, text: str) -> float:
        """
        Calculate fuzzy match score using Levenshtein distance.
        Returns score between 0 and 1.
        """
        if not query or not text:
            return 0.0
        
        # Simple Levenshtein distance implementation
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
    
    def _get_cell_context(self, cell_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get contextual information about a cell."""
        return {
            "table_headers": cell_data["table_headers"],
            "table_size": f"{cell_data['table_num_rows']}x{cell_data['table_num_cols']}",
            "position": f"({cell_data['row']}, {cell_data['col']})",
        }
    
    def search_by_column(self, column_name: str, value: str = None) -> List[SearchResult]:
        """
        Search for all cells in a specific column, optionally filtering by value.
        
        Args:
            column_name: Name of the column header
            value: Optional value to filter by
        """
        if value:
            return self.search(
                query=value,
                mode=SearchMode.CONTAINS,
                column=column_name,
            )
        else:
            return self.search(
                query="",
                mode=SearchMode.CONTAINS,
                search_headers=False,
                column=column_name,
            )
    
    def search_by_row(self, table_index: int, row_number: int) -> List[SearchResult]:
        """Get all cells in a specific row of a table."""
        return self.search(
            query="",
            mode=SearchMode.CONTAINS,
            search_headers=False,
            table_index=table_index,
            row=row_number,
        )
    
    def search_numeric_range(
        self,
        column: str,
        min_value: float = None,
        max_value: float = None,
    ) -> List[SearchResult]:
        """
        Search for numeric values within a range in a specific column.
        
        Args:
            column: Column header name
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
        """
        results = []
        
        for cell_data in self.cell_index:
            if column and cell_data["header"] != column:
                continue
            
            try:
                # Try to parse cell text as number
                cell_value = float(cell_data["text"].replace(",", ""))
                
                # Check range
                if min_value is not None and cell_value < min_value:
                    continue
                if max_value is not None and cell_value > max_value:
                    continue
                
                results.append(SearchResult(
                    table_index=cell_data["table_index"],
                    table_source=cell_data["table_source"],
                    row=cell_data["row"],
                    col=cell_data["col"],
                    header=cell_data["header"],
                    cell_text=cell_data["text"],
                    match_score=1.0,
                    context=self._get_cell_context(cell_data),
                ))
            except (ValueError, AttributeError):
                # Skip non-numeric cells
                continue
        
        return results
    
    def get_cell(self, table_index: int, row: int, col: int) -> Optional[SearchResult]:
        """Get a specific cell by table index, row, and column."""
        for cell_data in self.cell_index:
            if (
                cell_data["table_index"] == table_index
                and cell_data["row"] == row
                and cell_data["col"] == col
            ):
                return SearchResult(
                    table_index=cell_data["table_index"],
                    table_source=cell_data["table_source"],
                    row=cell_data["row"],
                    col=cell_data["col"],
                    header=cell_data["header"],
                    cell_text=cell_data["text"],
                    match_score=1.0,
                    context=self._get_cell_context(cell_data),
                )
        return None
    
    def get_table_summary(self, table_index: int) -> Dict[str, Any]:
        """Get summary information about a specific table."""
        table = self.tables[table_index]
        
        return {
            "index": table.get("index"),
            "source": table.get("source"),
            "num_rows": table.get("num_rows"),
            "num_columns": table.get("num_columns"),
            "headers": table.get("headers"),
            "total_cells": len(table.get("cells", [])),
        }
    
    def export_results_to_dict(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Export search results to a list of dictionaries."""
        return [
            {
                "table_index": r.table_index,
                "table_source": r.table_source,
                "row": r.row,
                "col": r.col,
                "header": r.header,
                "cell_text": r.cell_text,
                "match_score": r.match_score,
                "context": r.context,
            }
            for r in results
        ]


def print_search_results(results: List[SearchResult], max_display: int = 20):
    """Pretty print search results."""
    print(f"\n{'=' * 80}")
    print(f"Found {len(results)} results")
    print("=" * 80)
    
    for i, result in enumerate(results[:max_display]):
        print(f"\n[{i+1}] Table {result.table_index} ({result.table_source})")
        print(f"    Position: Row {result.row}, Col {result.col}")
        print(f"    Column: {result.header}")
        print(f"    Value: {result.cell_text}")
        print(f"    Match Score: {result.match_score:.2f}")
        if result.context:
            print(f"    Context: {result.context['position']} in {result.context['table_size']} table")
    
    if len(results) > max_display:
        print(f"\n... and {len(results) - max_display} more results")


# Example usage
if __name__ == "__main__":
    # Load table data from JSON
    with open('results/all_tables_output.json', 'r', encoding='utf-8') as f:
        tables_data = json.load(f)

    output_dir = "results"

    # Initialize search engine
    search_engine = TableSearchEngine(tables_data)
    
    print("=" * 80)
    print("TABLE SEARCH ENGINE - EXAMPLES")
    print("=" * 80)
    
    # Example 1: Simple text search
    print("\n1. Search for 'CONTRIBUTING SCIENTISTS' (contains mode):")
    results = search_engine.search("CONTRIBUTING SCIENTISTS", mode=SearchMode.CONTAINS)
    print_search_results(results)
    
    # Example 2: Exact match search
    print("\n2. Search for exact value 'CONTRIBUTING SCIENTISTS':")
    results = search_engine.search("CONTRIBUTING SCIENTISTS", mode=SearchMode.EXACT)
    print_search_results(results)
    
    # Example 3: Search in specific column
    print("\n3. Search in 'CONTRIBUTING SCIENTISTS' column:")
    results = search_engine.search_by_column("CONTRIBUTING SCIENTISTS")
    print_search_results(results)
    
    # Example 4: Search specific column with value
    print("\n4. Search for 'Study Pathologist' in 'CONTRIBUTING SCIENTISTS' column:")
    results = search_engine.search_by_column("CONTRIBUTING SCIENTISTS", "Study Pathologist")
    print_search_results(results)

    # Example 4.1: Search specific column with value
    print("\n4.1. Search for 'Study Pathologist' in 'CONTRIBUTING SCIENTISTS' column:")
    results = search_engine.search_by_column("CONTRIBUTING SCIENTISTS", "Pathology")
    print_search_results(results)
    
    # # Example 5: Get specific row
    # print("\n5. Get all cells from Table 0, Row 0:")
    # results = search_engine.search_by_row(table_index=0, row_number=0)
    # print_search_results(results)
    
    # # Example 6: Regex search
    # print("\n6. Search using regex pattern (\\d{4}):")
    # results = search_engine.search(r"\d{4}", mode=SearchMode.REGEX)
    # print_search_results(results)
    
    # # Example 7: Fuzzy search
    # print("\n7. Fuzzy search for 'PAUSANA' (should match 'PAUSANIA'):")
    # results = search_engine.search("PAUSANA", mode=SearchMode.FUZZY)
    # print_search_results(results)
    
    # # Example 8: Numeric range search
    # print("\n8. Search for prices between 100 and 600:")
    # results = search_engine.search_numeric_range("Price", min_value=100, max_value=600)
    # print_search_results(results)
    
    # # Example 9: Get specific cell
    # print("\n9. Get cell at Table 0, Row 1, Col 2:")
    # cell = search_engine.get_cell(table_index=0, row=1, col=2)
    # if cell:
    #     print(f"    Value: {cell.cell_text}")
    #     print(f"    Column: {cell.header}")
    
    # # Example 10: Get table summary
    # print("\n10. Table summaries:")
    # for i in range(len(tables_data)):
    #     summary = search_engine.get_table_summary(i)
    #     print(f"\nTable {i}:")
    #     print(f"    Source: {summary['source']}")
    #     print(f"    Size: {summary['num_rows']} rows × {summary['num_columns']} columns")
    #     print(f"    Headers: {summary['headers']}")
    
    # # Example 11: Export results to JSON
    # print("\n11. Export search results to JSON:")
    # results = search_engine.search("LED", mode=SearchMode.CONTAINS)
    # results_dict = search_engine.export_results_to_dict(results)
    # with open(f'{output_dir}/search_results.json', 'w', encoding='utf-8') as f:
    #     json.dump(results_dict, f, indent=2, ensure_ascii=False)
    # print(f"Saved to {output_dir}/search_results.json")