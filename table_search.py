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
    row: int
    col: int
    header: str
    header_levels: List[str]
    cell_text: str
    match_score: float = 1.0
    match_type: str = "cell"  # 'cell', 'header', 'header_level', 'title'
    matched_header_level: Optional[int] = None
    context: Optional[Dict[str, Any]] = None

class TableSearchEngine:
    """
    Advanced search engine for table data with cell-level granularity.
    Supports multi-row headers, grouped columns, and table titles.
    """
    
    def __init__(self, tables_data: List[Dict[str, Any]]):
        """
        Initialize search engine with table data.
        
        Args:
            tables_data: List of table dictionaries with enhanced structure
        """
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
            column_headers = table.get("column_headers", [])
            
            # Index table metadata
            self.table_index.append({
                "table_index": table_idx,
                "table_source": table_source,
                "table_title": table_title,
                "num_rows": table.get("num_rows", 0),
                "num_columns": table.get("num_columns", 0),
                "header_row_count": table.get("header_row_count", 1),
                "column_groups": table.get("column_groups", []),
            })
            
            # Index data cells
            for cell in table.get("cells", []):
                col_idx = cell.get("col", 0)
                
                # Get column header info
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
                    "row": cell.get("row"),
                    "col": col_idx,
                    "header": cell.get("header", ""),
                    "header_levels": header_levels,
                    "text": cell.get("text", ""),
                    "is_numeric": cell.get("is_numeric", False),
                    "table_headers": table.get("headers", []),
                    "table_num_rows": table.get("num_rows", 0),
                    "table_num_cols": table.get("num_columns", 0),
                    "table_header_row_count": table.get("header_row_count", 1),
                    "column_groups": table.get("column_groups", []),
                })
    
    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.CONTAINS,
        case_sensitive: bool = False,
        search_titles: bool = True,
        search_headers: bool = True,
        search_cells: bool = True,
        search_all_header_levels: bool = True,
        table_index: Optional[int] = None,
        table_title: Optional[str] = None,
        column: Optional[str] = None,
        column_group: Optional[str] = None,
        header_level: Optional[int] = None,
        row: Optional[int] = None,
        numeric_only: bool = False,
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search tables at cell level with support for hierarchical headers and titles.
        
        Args:
            query: Search query string
            mode: Search mode (exact, contains, starts_with, ends_with, regex, fuzzy)
            case_sensitive: Whether search is case-sensitive
            search_titles: Include table titles in search
            search_headers: Include header names in search
            search_cells: Include cell values in search
            search_all_header_levels: Search across all header levels
            table_index: Filter by specific table index
            table_title: Filter by table title (partial match)
            column: Filter by column header name (full path)
            column_group: Filter by column group parent name
            header_level: Search only in specific header level (0-based)
            row: Filter by specific row number
            numeric_only: Only search numeric cells
            max_results: Maximum number of results to return
        
        Returns:
            List of SearchResult objects
        """
        results = []
        
        # Search in table titles first
        if search_titles and query:
            for table_meta in self.table_index:
                if table_index is not None and table_meta["table_index"] != table_index:
                    continue
                
                if table_meta["table_title"]:
                    match_result = self._match_text(
                        query, table_meta["table_title"], mode, case_sensitive
                    )
                    if match_result["matched"]:
                        # Return a representative result for the table
                        results.append(SearchResult(
                            table_index=table_meta["table_index"],
                            table_source=table_meta["table_source"],
                            table_title=table_meta["table_title"],
                            row=-1,  # Special value for title match
                            col=-1,
                            header="",
                            header_levels=[],
                            cell_text="",
                            match_score=match_result["score"],
                            match_type="title",
                            context={
                                "table_size": f"{table_meta['num_rows']}x{table_meta['num_columns']}",
                                "header_row_count": table_meta["header_row_count"],
                            },
                        ))
        
        # Search in cells
        for cell_data in self.cell_index:
            # Apply filters
            if table_index is not None and cell_data["table_index"] != table_index:
                continue
            
            if table_title is not None and table_title.lower() not in cell_data["table_title"].lower():
                continue
            
            if column is not None and cell_data["header"] != column:
                continue
            
            if row is not None and cell_data["row"] != row:
                continue
            
            if numeric_only and not cell_data["is_numeric"]:
                continue
            
            # Filter by column group
            if column_group is not None:
                col_idx = cell_data["col"]
                in_group = False
                for group in cell_data["column_groups"]:
                    if group["parent"] == column_group and col_idx in group["columns"]:
                        in_group = True
                        break
                if not in_group:
                    continue
            
            # Search in cell text
            if search_cells and query:
                match_result = self._match_text(
                    query, cell_data["text"], mode, case_sensitive
                )
                if match_result["matched"]:
                    results.append(SearchResult(
                        table_index=cell_data["table_index"],
                        table_source=cell_data["table_source"],
                        table_title=cell_data["table_title"],
                        row=cell_data["row"],
                        col=cell_data["col"],
                        header=cell_data["header"],
                        header_levels=cell_data["header_levels"],
                        cell_text=cell_data["text"],
                        match_score=match_result["score"],
                        match_type="cell",
                        context=self._get_cell_context(cell_data),
                    ))
            
            # Search in header levels
            if search_headers and cell_data["header_levels"] and query:
                if search_all_header_levels:
                    # Search across all levels
                    for level_idx, level_text in enumerate(cell_data["header_levels"]):
                        if header_level is not None and level_idx != header_level:
                            continue
                        
                        match_result = self._match_text(
                            query, level_text, mode, case_sensitive
                        )
                        if match_result["matched"]:
                            results.append(SearchResult(
                                table_index=cell_data["table_index"],
                                table_source=cell_data["table_source"],
                                table_title=cell_data["table_title"],
                                row=cell_data["row"],
                                col=cell_data["col"],
                                header=cell_data["header"],
                                header_levels=cell_data["header_levels"],
                                cell_text=cell_data["text"],
                                match_score=match_result["score"],
                                match_type="header_level",
                                matched_header_level=level_idx,
                                context=self._get_cell_context(cell_data),
                            ))
                else:
                    # Search only in full header path
                    match_result = self._match_text(
                        query, cell_data["header"], mode, case_sensitive
                    )
                    if match_result["matched"]:
                        results.append(SearchResult(
                            table_index=cell_data["table_index"],
                            table_source=cell_data["table_source"],
                            table_title=cell_data["table_title"],
                            row=cell_data["row"],
                            col=cell_data["col"],
                            header=cell_data["header"],
                            header_levels=cell_data["header_levels"],
                            cell_text=cell_data["text"],
                            match_score=match_result["score"],
                            match_type="header",
                            context=self._get_cell_context(cell_data),
                        ))
        
        # Remove duplicates (same cell matched multiple times)
        seen = set()
        unique_results = []
        for r in results:
            key = (r.table_index, r.row, r.col, r.match_type)
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        # Sort by match score (descending), then by table/row/col
        unique_results.sort(
            key=lambda x: (-x.match_score, x.table_index, x.row if x.row >= 0 else 0, x.col if x.col >= 0 else 0)
        )
        
        # Apply max_results limit
        if max_results:
            unique_results = unique_results[:max_results]
        
        return unique_results
    
    def search_by_title(
        self,
        title_query: str,
        mode: SearchMode = SearchMode.CONTAINS
    ) -> List[int]:
        """
        Search for tables by title and return table indices.
        
        Args:
            title_query: Search query for table title
            mode: Search mode
        
        Returns:
            List of table indices that match
        """
        matching_tables = []
        
        for table_meta in self.table_index:
            if table_meta["table_title"]:
                match_result = self._match_text(
                    title_query, table_meta["table_title"], mode, False
                )
                if match_result["matched"]:
                    matching_tables.append(table_meta["table_index"])
        
        return matching_tables
    
    def get_table_by_title(
        self,
        title_query: str,
        mode: SearchMode = SearchMode.CONTAINS
    ) -> Optional[Dict[str, Any]]:
        """
        Get the first table that matches the title query.
        
        Args:
            title_query: Search query for table title
            mode: Search mode
        
        Returns:
            Table dict or None
        """
        matching_indices = self.search_by_title(title_query, mode)
        
        if matching_indices:
            return self.tables[matching_indices[0]]
        
        return None
    
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
            matched = score > 0.6
        
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
            "table_title": cell_data.get("table_title", ""),
            "table_headers": cell_data["table_headers"],
            "table_size": f"{cell_data['table_num_rows']}x{cell_data['table_num_cols']}",
            "position": f"({cell_data['row']}, {cell_data['col']})",
            "header_row_count": cell_data["table_header_row_count"],
            "column_groups": cell_data["column_groups"],
        }
    
    def search_by_column(
        self, 
        column_name: str, 
        value: str = None,
        mode: SearchMode = SearchMode.CONTAINS,
        table_title: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for all cells in a specific column, optionally filtering by value and table title.
        
        Args:
            column_name: Name of the column header (full path or any level)
            value: Optional value to filter by
            mode: Search mode for value matching
            table_title: Optional table title filter
        """
        results = []
        
        for cell_data in self.cell_index:
            # Filter by table title
            if table_title is not None and table_title.lower() not in cell_data["table_title"].lower():
                continue
            
            # Check if column_name matches any header level
            matches_column = False
            if cell_data["header"] == column_name:
                matches_column = True
            elif column_name in cell_data["header_levels"]:
                matches_column = True
            
            if not matches_column:
                continue
            
            # If value filter is specified, check it
            if value is not None:
                match_result = self._match_text(value, cell_data["text"], mode, False)
                if not match_result["matched"]:
                    continue
                score = match_result["score"]
            else:
                score = 1.0
            
            results.append(SearchResult(
                table_index=cell_data["table_index"],
                table_source=cell_data["table_source"],
                table_title=cell_data["table_title"],
                row=cell_data["row"],
                col=cell_data["col"],
                header=cell_data["header"],
                header_levels=cell_data["header_levels"],
                cell_text=cell_data["text"],
                match_score=score,
                match_type="column_filter",
                context=self._get_cell_context(cell_data),
            ))
        
        return results
    
    def search_by_column_group(
        self, 
        group_name: str, 
        value: str = None,
        table_title: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for all cells under a column group.
        
        Args:
            group_name: Parent name of the column group
            value: Optional value to filter by
            table_title: Optional table title filter
        """
        return self.search(
            query=value if value else "",
            mode=SearchMode.CONTAINS,
            search_titles=False,
            search_headers=False,
            search_cells=True if value else False,
            column_group=group_name,
            table_title=table_title,
        )
    
    def search_by_row(
        self, 
        table_index: int, 
        row_number: int
    ) -> List[SearchResult]:
        """Get all cells in a specific row of a table."""
        results = []
        
        for cell_data in self.cell_index:
            if cell_data["table_index"] == table_index and cell_data["row"] == row_number:
                results.append(SearchResult(
                    table_index=cell_data["table_index"],
                    table_source=cell_data["table_source"],
                    table_title=cell_data["table_title"],
                    row=cell_data["row"],
                    col=cell_data["col"],
                    header=cell_data["header"],
                    header_levels=cell_data["header_levels"],
                    cell_text=cell_data["text"],
                    match_score=1.0,
                    match_type="row_filter",
                    context=self._get_cell_context(cell_data),
                ))
        
        return results
    
    def search_by_header_level(
        self,
        query: str,
        level: int,
        mode: SearchMode = SearchMode.CONTAINS,
        table_title: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search in a specific header level.
        
        Args:
            query: Search query
            level: Header level (0 = top level, 1 = second level, etc.)
            mode: Search mode
            table_title: Optional table title filter
        """
        return self.search(
            query=query,
            mode=mode,
            search_titles=False,
            search_headers=True,
            search_cells=False,
            search_all_header_levels=True,
            header_level=level,
            table_title=table_title,
        )
    
    def search_numeric_range(
        self,
        column: Optional[str] = None,
        column_group: Optional[str] = None,
        table_title: Optional[str] = None,
        min_value: float = None,
        max_value: float = None,
    ) -> List[SearchResult]:
        """
        Search for numeric values within a range.
        
        Args:
            column: Column header name (optional)
            column_group: Column group name (optional)
            table_title: Table title filter (optional)
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
        """
        results = []
        
        for cell_data in self.cell_index:
            # Filter by table title
            if table_title is not None and table_title.lower() not in cell_data["table_title"].lower():
                continue
            
            # Apply column filter
            if column is not None:
                matches_column = (
                    cell_data["header"] == column or 
                    column in cell_data["header_levels"]
                )
                if not matches_column:
                    continue
            
            # Apply column group filter
            if column_group is not None:
                col_idx = cell_data["col"]
                in_group = False
                for group in cell_data["column_groups"]:
                    if group["parent"] == column_group and col_idx in group["columns"]:
                        in_group = True
                        break
                if not in_group:
                    continue
            
            # Try to parse as number
            try:
                cell_value = float(cell_data["text"].replace(",", "").replace("$", "").strip())
                
                # Check range
                if min_value is not None and cell_value < min_value:
                    continue
                if max_value is not None and cell_value > max_value:
                    continue
                
                results.append(SearchResult(
                    table_index=cell_data["table_index"],
                    table_source=cell_data["table_source"],
                    table_title=cell_data["table_title"],
                    row=cell_data["row"],
                    col=cell_data["col"],
                    header=cell_data["header"],
                    header_levels=cell_data["header_levels"],
                    cell_text=cell_data["text"],
                    match_score=1.0,
                    match_type="numeric_range",
                    context=self._get_cell_context(cell_data),
                ))
            except (ValueError, AttributeError):
                continue
        
        return results
    
    def get_cell(
        self, 
        table_index: int, 
        row: int, 
        col: int
    ) -> Optional[SearchResult]:
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
                    table_title=cell_data["table_title"],
                    row=cell_data["row"],
                    col=cell_data["col"],
                    header=cell_data["header"],
                    header_levels=cell_data["header_levels"],
                    cell_text=cell_data["text"],
                    match_score=1.0,
                    match_type="direct_access",
                    context=self._get_cell_context(cell_data),
                )
        return None
    
    def get_column_info(self, table_index: int) -> List[Dict[str, Any]]:
        """Get detailed column information for a table."""
        table = self.tables[table_index]
        return table.get("column_headers", [])
    
    def get_column_groups(self, table_index: int) -> List[Dict[str, Any]]:
        """Get column group information for a table."""
        table = self.tables[table_index]
        return table.get("column_groups", [])
    
    def get_table_summary(self, table_index: int) -> Dict[str, Any]:
        """Get summary information about a specific table."""
        table = self.tables[table_index]
        
        return {
            "index": table.get("index"),
            "source": table.get("source"),
            "title": table.get("title", ""),
            "num_rows": table.get("num_rows"),
            "num_columns": table.get("num_columns"),
            "header_row_count": table.get("header_row_count", 1),
            "headers": table.get("headers"),
            "column_headers": table.get("column_headers", []),
            "column_groups": table.get("column_groups", []),
            "total_cells": len(table.get("cells", [])),
        }
    
    def list_all_tables(self) -> List[Dict[str, Any]]:
        """List all tables with their titles and basic info."""
        return [
            {
                "index": t["table_index"],
                "title": t["table_title"],
                "source": t["table_source"],
                "size": f"{t['num_rows']}x{t['num_columns']}",
            }
            for t in self.table_index
        ]
    
    def export_results_to_dict(
        self, 
        results: List[SearchResult]
    ) -> List[Dict[str, Any]]:
        """Export search results to a list of dictionaries."""
        return [
            {
                "table_index": r.table_index,
                "table_source": r.table_source,
                "table_title": r.table_title,
                "row": r.row,
                "col": r.col,
                "header": r.header,
                "header_levels": r.header_levels,
                "cell_text": r.cell_text,
                "match_score": r.match_score,
                "match_type": r.match_type,
                "matched_header_level": r.matched_header_level,
                "context": r.context,
            }
            for r in results
        ]

    def get_row_data(
        self, 
        table_index: int, 
        row_number: int,
        as_dict: bool = True
    ) -> Union[Dict[str, str], List[str]]:
        """
        Get all data from a specific row.
        
        Args:
            table_index: Table index
            row_number: Row number
            as_dict: If True, return as {header: value} dict, else as list
        
        Returns:
            Dict mapping headers to values, or list of values
        """
        row_cells = self.search_by_row(table_index, row_number)
        
        if not row_cells:
            return {} if as_dict else []
        
        # Sort by column index
        row_cells.sort(key=lambda x: x.col)
        
        if as_dict:
            return {cell.header: cell.cell_text for cell in row_cells}
        else:
            return [cell.cell_text for cell in row_cells]

    def get_cell_value_by_header(
        self,
        table_index: int,
        row_number: int,
        header_name: str
    ) -> Optional[str]:
        """
        Get the value of a specific cell by table, row, and column header name.
        
        Args:
            table_index: Table index
            row_number: Row number
            header_name: Column header name (can be full path or any level)
        
        Returns:
            Cell value or None if not found
        """
        for cell_data in self.cell_index:
            if (
                cell_data["table_index"] == table_index
                and cell_data["row"] == row_number
            ):
                # Check if header matches
                if (
                    cell_data["header"] == header_name
                    or header_name in cell_data["header_levels"]
                ):
                    return cell_data["text"]
        
        return None

    def search_with_related_columns(
        self,
        query: str,
        search_column: Optional[str] = None,
        return_columns: Optional[List[str]] = None,
        mode: SearchMode = SearchMode.CONTAINS,
        table_index: Optional[int] = None,
        table_title: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search in one column and return values from related columns in the same row.
        
        Args:
            query: Search query
            search_column: Column to search in (if None, searches all columns)
            return_columns: List of column headers to return values from
                        (if None, returns all columns)
            mode: Search mode
            table_index: Filter by table index
            table_title: Filter by table title
            max_results: Maximum results to return
        
        Returns:
            List of dicts with matched row data
        """
        # First, search for matching cells
        if search_column:
            results = self.search_by_column(
                column_name=search_column,
                value=query,
                mode=mode,
                table_title=table_title
            )
        else:
            results = self.search(
                query=query,
                mode=mode,
                table_index=table_index,
                table_title=table_title,
                search_cells=True,
                search_headers=False,
                search_titles=False,
            )
        
        if table_index is not None:
            results = [r for r in results if r.table_index == table_index]
        
        # For each result, get the full row data
        enriched_results = []
        
        for result in results:
            if result.row < 0:  # Skip title matches
                continue
            
            # Get full row data
            row_data = self.get_row_data(result.table_index, result.row, as_dict=True)
            
            # Filter to requested columns if specified
            if return_columns:
                filtered_data = {}
                for col_name in return_columns:
                    # Try to find the column in row_data
                    value = None
                    for header, val in row_data.items():
                        if header == col_name or col_name in header:
                            value = val
                            break
                    filtered_data[col_name] = value
                row_data = filtered_data
            
            enriched_results.append({
                "table_index": result.table_index,
                "table_title": result.table_title,
                "table_source": result.table_source,
                "row": result.row,
                "matched_column": result.header,
                "matched_value": result.cell_text,
                "match_score": result.match_score,
                "row_data": row_data,
            })
        
        if max_results:
            enriched_results = enriched_results[:max_results]
        
        return enriched_results

    def search_and_get_column_value(
        self,
        query: str,
        search_column: str,
        return_column: str,
        mode: SearchMode = SearchMode.CONTAINS,
        table_index: Optional[int] = None,
        table_title: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search in one column and return the corresponding value from another column.
        
        This is a convenience method for the common use case of:
        "Find rows where column A contains X, and give me the value from column B"
        
        Args:
            query: Search query
            search_column: Column to search in
            return_column: Column to return value from
            mode: Search mode
            table_index: Filter by table index
            table_title: Filter by table title
        
        Returns:
            List of dicts with search match and corresponding value
        
        Example:
            # Find "Pathology" in "Role" column, return "Details" value
            results = search_engine.search_and_get_column_value(
                query="Pathology",
                search_column="Role",
                return_column="Details",
                table_title="CONTRIBUTING SCIENTISTS"
            )
        """
        # Search in the specified column
        search_results = self.search_by_column(
            column_name=search_column,
            value=query,
            mode=mode,
            table_title=table_title
        )
        
        if table_index is not None:
            search_results = [r for r in search_results if r.table_index == table_index]
        
        # For each match, get the corresponding value from return_column
        results = []
        
        for result in search_results:
            if result.row < 0:  # Skip title matches
                continue
            
            # Get the value from the return column
            return_value = self.get_cell_value_by_header(
                table_index=result.table_index,
                row_number=result.row,
                header_name=return_column
            )
            
            results.append({
                "table_index": result.table_index,
                "table_title": result.table_title,
                "table_source": result.table_source,
                "row": result.row,
                "search_column": search_column,
                "search_value": result.cell_text,
                "return_column": return_column,
                "return_value": return_value,
                "match_score": result.match_score,
            })
        
        return results

    def search_and_get_row(
        self,
        query: str,
        mode: SearchMode = SearchMode.CONTAINS,
        table_index: Optional[int] = None,
        table_title: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for a query and return the entire row data for each match.
        This doesn't assume any column names.
        
        Args:
            query: Search query
            mode: Search mode
            table_index: Filter by table index
            table_title: Filter by table title
        
        Returns:
            List of dicts with full row data
        """
        # Search for the query
        search_results = self.search(
            query=query,
            mode=mode,
            table_index=table_index,
            table_title=table_title,
            search_cells=True,
            search_headers=False,
            search_titles=False,
        )
        
        # For each match, get the full row
        results = []
        seen_rows = set()  # Avoid duplicates
        
        for result in search_results:
            if result.row < 0:  # Skip title matches
                continue
            
            row_key = (result.table_index, result.row)
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            
            # Get all cells in this row
            row_cells = [
                c for c in self.cell_index 
                if c['table_index'] == result.table_index and c['row'] == result.row
            ]
            
            # Sort by column
            row_cells.sort(key=lambda x: x['col'])
            
            # Build row data
            row_data = {}
            for cell in row_cells:
                header = cell.get('header', f"Column_{cell['col']}")
                row_data[header] = cell.get('text', '')
            
            results.append({
                "table_index": result.table_index,
                "table_title": result.table_title,
                "table_source": result.table_source,
                "row": result.row,
                "matched_in_column": result.header,
                "matched_value": result.cell_text,
                "match_score": result.match_score,
                "row_data": row_data,
                "all_values": [cell.get('text', '') for cell in row_cells],
            })
        
        return results

    def find_table_index_by_title(self, title_query: str) -> int | None:
        """
        Find the table index whose title contains title_query (case-insensitive).
        """
        q = title_query.lower()
        for t in self.tables:
            title = (t.get("title") or "").lower()
            if q in title:
                return t["index"]
        return None
    
    def get_table_rows_as_lists(self, table_index: int) -> list[list[str]]:
        """
        Return table rows as a list of lists of cell text, in column order.
        """
        table = self.tables[table_index]
        n_cols = table.get("num_columns", 0)
        n_rows = table.get("num_rows", 0)
        rows = [[""] * n_cols for _ in range(n_rows)]
        for cell in table.get("cells", []):
            r = cell["row"]
            c = cell["col"]
            if 0 <= r < n_rows and 0 <= c < n_cols:
                rows[r][c] = cell.get("text", "")
        return rows
    
    def search_roles_and_values_in_table(self, json_file, table_name, role_match=None, value_match=None):
        """
        Search for roles and values in a specific table from the extracted JSON.
        
        Args:
            json_file: Path to the JSON file containing extracted tables
            table_name: Name of the table to search in
            role_match: Role/column to search for (e.g., "Clinical Assessment", "Pathology")
            value_match: Value to search for in any cell
        
        Returns:
            List of matching results with context
        """
        import json
        
        # Load the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        
        # Find the specified table
        for table in data.get('tables', []):
            if table.get('title', '').strip().lower() == table_name.strip().lower():
                headers = table.get('headers', [])
                rows = table.get('rows', [])
                
                # Special handling for CONTRIBUTING SCIENTISTS table
                if 'contributing scientists' in table_name.lower():
                    results.extend(self.search_contributing_scientists_table(
                        table, role_match
                    ))
                else:
                    # Standard table search
                    results.extend(self.search_standard_table(
                        table, headers, rows, role_match
                    ))
        
        return results

    def search_contributing_scientists_table(self, table, role_match=None, value_match=None):
        """
        Special search logic for the CONTRIBUTING SCIENTISTS table with multi-level headers.
        """
        results = []
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        if not headers or not rows:
            return results
        
        # Build role-to-column mapping from headers
        # Headers structure: [level0_headers, level1_headers, ...]
        role_columns = {}
        
        if len(headers) >= 2:
            # Level 0: Main categories (e.g., "Clinical Assessment", "Pathology")
            # Level 1: Subcategories (e.g., "In-Life", "Necropsy")
            level0 = headers[0]
            level1 = headers[1] if len(headers) > 1 else []
            
            # Map each level1 column to its parent level0 role
            col_idx = 0
            for h0 in level0:
                role_name = h0.get('text', '').strip()
                colspan = h0.get('colspan', 1)
                
                # This role spans multiple columns
                for i in range(colspan):
                    if col_idx + i < len(level1):
                        subcol_name = level1[col_idx + i].get('text', '').strip()
                        role_columns[col_idx + i] = {
                            'role': role_name,
                            'subcategory': subcol_name
                        }
                
                col_idx += colspan
        
        # Search based on role_match
        if role_match:
            role_match_lower = role_match.strip().lower()
            
            # Find columns that match the role
            matching_columns = []
            for col_idx, role_info in role_columns.items():
                if role_match_lower in role_info['role'].lower():
                    matching_columns.append(col_idx)
            
            # Extract data from matching columns
            for row_idx, row in enumerate(rows):
                cells = row.get('cells', [])
                
                for col_idx in matching_columns:
                    if col_idx < len(cells):
                        cell_value = cells[col_idx].strip()
                        
                        # Skip empty cells
                        if not cell_value or cell_value == '-':
                            continue
                        
                        # Get the scientist name (usually first column)
                        scientist_name = cells[0].strip() if len(cells) > 0 else "Unknown"
                        
                        result = {
                            'table': table.get('title', ''),
                            'row_index': row_idx,
                            'role': role_columns[col_idx]['role'],
                            'subcategory': role_columns[col_idx]['subcategory'],
                            'scientist': scientist_name,
                            'value': cell_value,
                            'full_row': cells
                        }
                        results.append(result)
        
        # Search based on value_match
        if value_match:
            value_match_lower = value_match.strip().lower()
            
            for row_idx, row in enumerate(rows):
                cells = row.get('cells', [])
                
                for col_idx, cell_value in enumerate(cells):
                    if value_match_lower in cell_value.strip().lower():
                        # Get role information for this column
                        role_info = role_columns.get(col_idx, {
                            'role': 'Unknown',
                            'subcategory': 'Unknown'
                        })
                        
                        scientist_name = cells[0].strip() if len(cells) > 0 else "Unknown"
                        
                        result = {
                            'table': table.get('title', ''),
                            'row_index': row_idx,
                            'role': role_info.get('role', 'Unknown'),
                            'subcategory': role_info.get('subcategory', 'Unknown'),
                            'scientist': scientist_name,
                            'value': cell_value.strip(),
                            'column_index': col_idx,
                            'full_row': cells
                        }
                        results.append(result)
        
        return results

    def search_standard_table(self, table, headers, rows, role_match=None, value_match=None):
        """
        Standard search logic for tables with simple header structure.
        """
        results = []
        
        # Flatten headers if multi-level
        flat_headers = []
        if headers:
            if isinstance(headers[0], list):
                # Multi-level headers - use the last level
                flat_headers = [h.get('text', '') for h in headers[-1]]
            elif isinstance(headers[0], dict):
                flat_headers = [h.get('text', '') for h in headers]
            else:
                flat_headers = headers
        
        # Search based on role_match (column name)
        if role_match:
            role_match_lower = role_match.strip().lower()
            
            # Find matching column indices
            matching_cols = []
            for idx, header in enumerate(flat_headers):
                if role_match_lower in header.lower():
                    matching_cols.append(idx)
            
            # Extract data from matching columns
            for row_idx, row in enumerate(rows):
                cells = row.get('cells', [])
                
                for col_idx in matching_cols:
                    if col_idx < len(cells):
                        cell_value = cells[col_idx].strip()
                        
                        if not cell_value:
                            continue
                        
                        result = {
                            'table': table.get('title', ''),
                            'row_index': row_idx,
                            'column': flat_headers[col_idx] if col_idx < len(flat_headers) else f"Column {col_idx}",
                            'value': cell_value,
                            'full_row': cells
                        }
                        results.append(result)
        
        # Search based on value_match
        if value_match:
            value_match_lower = value_match.strip().lower()
            
            for row_idx, row in enumerate(rows):
                cells = row.get('cells', [])
                
                for col_idx, cell_value in enumerate(cells):
                    if value_match_lower in cell_value.strip().lower():
                        result = {
                            'table': table.get('title', ''),
                            'row_index': row_idx,
                            'column': flat_headers[col_idx] if col_idx < len(flat_headers) else f"Column {col_idx}",
                            'value': cell_value.strip(),
                            'column_index': col_idx,
                            'full_row': cells
                        }
                        results.append(result)
        
        return results

def print_search_results(results: List[SearchResult], max_display: int = 20):
    """Pretty print search results."""
    print(f"\n{'=' * 80}")
    print(f"Found {len(results)} results")
    print("=" * 80)
    
    for i, result in enumerate(results[:max_display]):
        print(f"\n[{i+1}] Table {result.table_index} ({result.table_source})")
        if result.table_title:
            print(f"    Table Title: {result.table_title}")
        
        if result.match_type == "title":
            print(f"    Match Type: Title Match")
            print(f"    Match Score: {result.match_score:.2f}")
        else:
            print(f"    Position: Row {result.row}, Col {result.col}")
            print(f"    Column: {result.header}")
            if result.header_levels:
                levels_str = " → ".join([f"'{l}'" for l in result.header_levels if l])
                print(f"    Header Hierarchy: {levels_str}")
            print(f"    Value: {result.cell_text}")
            print(f"    Match Type: {result.match_type}")
            if result.matched_header_level is not None:
                print(f"    Matched Header Level: {result.matched_header_level}")
            print(f"    Match Score: {result.match_score:.2f}")
            if result.context:
                print(f"    Context: {result.context['position']} in {result.context['table_size']} table")
    
    if len(results) > max_display:
        print(f"\n... and {len(results) - max_display} more results")

def print_enriched_results(results: List[Dict[str, Any]], max_display: int = 20):
    """Pretty print enriched search results with full row data."""
    print(f"\n{'=' * 80}")
    print(f"Found {len(results)} results")
    print("=" * 80)
    
    for i, result in enumerate(results[:max_display]):
        print(f"\n[{i+1}] Table {result['table_index']} ({result['table_source']})")
        if result.get('table_title'):
            print(f"    Table Title: {result['table_title']}")
        print(f"    Row: {result['row']}")
        
        if 'matched_column' in result:
            print(f"    Matched in column: {result['matched_column']}")
            print(f"    Matched value: {result['matched_value']}")
            print(f"    Match Score: {result['match_score']:.2f}")
            print(f"\n    Full Row Data:")
            for header, value in result['row_data'].items():
                print(f"      {header}: {value}")
        
        elif 'search_column' in result:
            print(f"    Search Column: {result['search_column']}")
            print(f"    Search Value: {result['search_value']}")
            print(f"    Return Column: {result['return_column']}")
            print(f"    Return Value: {result['return_value']}")
            print(f"    Match Score: {result['match_score']:.2f}")
    
    if len(results) > max_display:
        print(f"\n... and {len(results) - max_display} more results")

# Example usage and comprehensive tests
if __name__ == "__main__":
    output_dir = "./results"
    # Load table data from JSON
    with open(f'{output_dir}/all_tables_output.json', 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    
    # Initialize search engine
    search_engine = TableSearchEngine(tables_data)
    
    print("=" * 80)
    print("ADVANCED TABLE SEARCH ENGINE - WITH ROW VALUE RETRIEVAL")
    print("=" * 80)

    results = search_engine.search_roles_and_values_in_table(
        f'{output_dir}/all_tables_output.json',
        'CONTRIBUTING SCIENTISTS',
        role_match='Clinical Assessment'
    )
    
    print(f"Found {len(results)} results for Clinical Assessment:")
    for result in results:
        print(f"\nScientist: {result['scientist']}")
        print(f"Role: {result['role']}")
        print(f"Subcategory: {result['subcategory']}")
        print(f"Value: {result['value']}")
    
    print("\n" + "="*80 + "\n")
    
    # Search for Pathology role
    results = search_engine.search_roles_and_values_in_table(
        f'{output_dir}/all_tables_output.json',
        'CONTRIBUTING SCIENTISTS',
        role_match='Pathology'
    )
    
    print(f"Found {len(results)} results for Pathology:")
    for result in results:
        print(f"\nScientist: {result['scientist']}")
        print(f"Role: {result['role']}")
        print(f"Subcategory: {result['subcategory']}")
        print(f"Value: {result['value']}")

    # # Method 1: Get just the Details value for "Pathology"
    # results = search_engine.search_and_get_column_value(
    #     query="Pathology",
    #     search_column="Pathology",
    #     return_column="Pathology",
    #     table_title="**3 CONTRIBUTING SCIENTISTS**"
    # )

    # for r in results:
    #     print(f"Pathology: {r['search_value']}")
    #     print(f"Pathology: {r['return_value']}")

    # # Method 2: Get entire row data
    # results = search_engine.search_with_related_columns(
    #     query="Pathology",
    #     search_column="Pathology",
    #     table_title="**3 CONTRIBUTING SCIENTISTS**"
    # )

    # for r in results:
    #     print(f"\nFound '{r['matched_value']}' in row {r['row']}")
    #     print("Full row data:")
    #     for header, value in r['row_data'].items():
    #         print(f"  {header}: {value}")
    
    # # Example 1: List all tables
    # print("\n" + "=" * 80)
    # print("1. List all tables:")
    # print("=" * 80)
    # all_tables = search_engine.list_all_tables()
    # for t in all_tables:
    #     print(f"  Table {t['index']}: '{t['title']}' ({t['source']}, {t['size']})")
    
    # # Example 2: Search for "Pathology" and get the Details value
    # print("\n" + "=" * 80)
    # print("2. Search for 'Pathology' in Role column, return Details value:")
    # print("=" * 80)
    # results = search_engine.search_and_get_column_value(
    #     query="Pathology",
    #     search_column="Role",
    #     return_column="Details",
    #     table_title="CONTRIBUTING SCIENTISTS"
    # )
    # print_enriched_results(results)
    
    # # Example 3: Search for "Pathology" and get entire row
    # print("\n" + "=" * 80)
    # print("3. Search for 'Pathology' and get entire row data:")
    # print("=" * 80)
    # results = search_engine.search_with_related_columns(
    #     query="Pathology",
    #     search_column="Role",
    #     table_title="CONTRIBUTING SCIENTISTS"
    # )
    # print_enriched_results(results)
    
    # # Example 4: Get all rows with "Pathology" in any column
    # print("\n" + "=" * 80)
    # print("4. Search for 'Pathology' in any column, return full rows:")
    # print("=" * 80)
    # results = search_engine.search_with_related_columns(
    #     query="Pathology",
    #     table_title="CONTRIBUTING SCIENTISTS"
    # )
    # print_enriched_results(results)
    
    # # Example 5: Get specific row data by index
    # print("\n" + "=" * 80)
    # print("5. Get row 3 data from table 0:")
    # print("=" * 80)
    # row_data = search_engine.get_row_data(table_index=0, row_number=3)
    # print("    Row data as dict:")
    # for header, value in row_data.items():
    #     print(f"      {header}: {value}")
    
    # # Example 6: Search "Director" and return specific columns
    # print("\n" + "=" * 80)
    # print("6. Search for 'Director', return only Role and Details columns:")
    # print("=" * 80)
    # results = search_engine.search_with_related_columns(
    #     query="Director",
    #     return_columns=["Role", "Details"],
    #     table_title="CONTRIBUTING SCIENTISTS"
    # )
    # print_enriched_results(results)
    
    # # Example 7: Multiple searches with column values
    # print("\n" + "=" * 80)
    # print("7. Find all roles containing 'Pathology' and their details:")
    # print("=" * 80)
    # results = search_engine.search_and_get_column_value(
    #     query="Pathology",
    #     search_column="Role",
    #     return_column="Details",
    #     mode=SearchMode.CONTAINS
    # )
    # for i, r in enumerate(results):
    #     print(f"\n  [{i+1}] {r['search_value']} → {r['return_value']}")
    
    # # Example 8: Get specific cell value
    # print("\n" + "=" * 80)
    # print("8. Get 'Details' value from row 3 of table 0:")
    # print("=" * 80)
    # details_value = search_engine.get_cell_value_by_header(
    #     table_index=0,
    #     row_number=3,
    #     header_name="Details"
    # )
    # print(f"    Details: {details_value}")
    
    # # Example 9: Export enriched results
    # print("\n" + "=" * 80)
    # print("9. Export enriched search results:")
    # print("=" * 80)
    # results = search_engine.search_with_related_columns(
    #     query="Pathology",
    #     table_title="CONTRIBUTING SCIENTISTS",
    #     max_results=10
    # )
    # with open('enriched_search_results.json', 'w', encoding='utf-8') as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)
    # print("    ✓ Saved to enriched_search_results.json")
    
    # print("\n" + "=" * 80)
    # print("SEARCH ENGINE READY")
    # print("=" * 80)