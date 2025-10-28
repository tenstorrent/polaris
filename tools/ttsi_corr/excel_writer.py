# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Excel Writer Module - Generate Excel reports with formatted tables.

This module handles creation of Excel workbooks with correlation results,
including formatting, formulas, and styling.

Responsibilities:
-----------------
- Create Excel workbooks with correlation data
- Apply number formatting to cells
- Add Excel formulas for calculated fields
- Apply borders and styling
- Configure frozen panes and filters
- Export data to CSV format

Classes:
--------
- ExcelFormatter: Utilities for Excel formatting and layout
  - col_letter(): Convert column index to Excel letter (A, B, ..., AA, AB)
  - apply_number_formats(): Apply number formatting to cells
  - apply_borders(): Apply borders to all cells
  - apply_freeze_panes(): Configure frozen panes

Functions:
----------
- write_csv() - Export data to CSV format

Migration Status:
-----------------
🟢 PARTIALLY IMPLEMENTED - Phase 2 in progress
  ✅ ExcelFormatter class created
  ✅ col_letter() static method implemented
  ✅ write_csv() function migrated
  🔴 write_correlation_xlsx() still in run_ttsi_corr.py (will migrate in later phase)

Usage:
------
    from ttsi_corr.excel_writer import ExcelFormatter, write_csv
    
    # Use ExcelFormatter utilities
    formatter = ExcelFormatter()
    col_name = formatter.col_letter(1)  # Returns 'A'
    
    # Write CSV
    write_csv(
        comparison=correlation_data,
        output_path=Path('output/correlation_result.csv')
    )

See Also:
---------
- tools/run_ttsi_corr.py: Main correlation script
- tools/ttsi_corr/chart_builder.py: Chart generation
- openpyxl: Excel library used for implementation
"""

import csv
from pathlib import Path
from typing import Any

from loguru import logger
from openpyxl.styles import Border, Side  # type: ignore[import-untyped]
from openpyxl.worksheet.worksheet import Worksheet  # type: ignore[import-untyped]


class ExcelFormatter:
    """
    Utilities for Excel formatting and layout.
    
    This class provides reusable Excel formatting utilities that can be
    used across multiple tools in the Polaris ecosystem. All methods are
    designed to be composable and testable.
    
    Static Methods:
    ---------------
    - col_letter(col_idx): Convert column index to Excel letter (A, B, ..., AA, AB)
    
    Instance Methods:
    -----------------
    - apply_number_formats(): Apply number formatting to worksheet cells
    - apply_borders(): Apply borders to cell range
    - apply_freeze_panes(): Configure frozen panes
    
    Example:
    --------
        >>> formatter = ExcelFormatter()
        >>> formatter.col_letter(1)
        'A'
        >>> formatter.col_letter(27)
        'AA'
        >>> formatter.col_letter(702)
        'ZZ'
    """
    
    @staticmethod
    def col_letter(col_idx: int) -> str:
        """
        Convert 1-based column index to Excel column letter.
        
        Converts numeric column indices to Excel-style column letters.
        Supports any positive column index.
        
        Args:
            col_idx: 1-based column index (1 for 'A', 2 for 'B', etc.)
            
        Returns:
            Excel column letter string (e.g., 'A', 'B', ..., 'Z', 'AA', 'AB', ...)
            
        Examples:
            >>> ExcelFormatter.col_letter(1)
            'A'
            >>> ExcelFormatter.col_letter(26)
            'Z'
            >>> ExcelFormatter.col_letter(27)
            'AA'
            >>> ExcelFormatter.col_letter(702)
            'ZZ'
            >>> ExcelFormatter.col_letter(703)
            'AAA'
        
        Note:
            This is a static method and can be called without instantiating
            the class: ExcelFormatter.col_letter(5)
        """
        result = ''
        while col_idx > 0:
            col_idx -= 1
            result = chr(65 + (col_idx % 26)) + result
            col_idx //= 26
        return result
    
    def apply_number_formats(
        self,
        ws: Worksheet,
        headers: list[str],
        comparison: list[dict[str, Any]]
    ) -> None:
        """
        Apply number formatting to worksheet cells based on data types.
        
        Formats cells according to their content:
        - Batch column: Integer format (0 decimal places)
        - Float values: Comma-separated with 2 decimal places (#,##0.00)
        - Other values: Default formatting
        
        Args:
            ws: Worksheet object to format
            headers: List of column headers
            comparison: List of data dictionaries (for type detection)
            
        Side Effects:
            Modifies cell number_format properties in the worksheet
            
        Example:
            >>> formatter = ExcelFormatter()
            >>> formatter.apply_number_formats(ws, headers, data)
        """
        for col_idx, header in enumerate(headers, start=1):
            col_letter_str = self.col_letter(col_idx)
            
            for row_idx in range(2, len(comparison) + 2):  # Start from row 2 (after header)
                cell = ws[f'{col_letter_str}{row_idx}']
                data_row_idx = row_idx - 2  # Convert back to 0-based index
                
                # Get the original value from comparison data
                if data_row_idx < len(comparison):
                    original_value = comparison[data_row_idx][header]
                    
                    # Apply number formatting based on column type
                    if header == 'Batch':
                        # Format Batch as integer with 0 decimal places
                        if isinstance(original_value, (int, float)):
                            cell.number_format = '0'
                    elif isinstance(original_value, float):
                        # Apply comma number format to all other numeric fields (floats)
                        cell.number_format = '#,##0.00'
    
    def apply_borders(
        self,
        ws: Worksheet,
        num_rows: int,
        num_cols: int
    ) -> None:
        """
        Apply thin borders to all cells in a range.
        
        Creates a consistent border style around all cells in the specified range,
        giving the table a professional, structured appearance.
        
        Args:
            ws: Worksheet object to format
            num_rows: Number of rows (including header)
            num_cols: Number of columns
            
        Side Effects:
            Modifies cell border properties in the worksheet
            
        Example:
            >>> formatter = ExcelFormatter()
            >>> formatter.apply_borders(ws, num_rows=100, num_cols=12)
        """
        # Create thin border style
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Apply borders to all cells
        for row in ws.iter_rows(min_row=1, max_row=num_rows, min_col=1, max_col=num_cols):
            for cell in row:
                cell.border = thin_border
    
    def apply_freeze_panes(
        self,
        ws: Worksheet,
        freeze_col: str,
        freeze_row: int = 2
    ) -> None:
        """
        Configure frozen panes for easier navigation.
        
        Freezes the header row and columns up to the specified column,
        allowing users to scroll while keeping headers visible.
        
        Args:
            ws: Worksheet object to configure
            freeze_col: Excel column letter to freeze after (e.g., 'E')
            freeze_row: Row number to freeze after (default: 2, freezes header)
            
        Side Effects:
            Sets the freeze_panes property on the worksheet
            
        Example:
            >>> formatter = ExcelFormatter()
            >>> formatter.apply_freeze_panes(ws, freeze_col='E', freeze_row=2)
            # Freezes columns A-D and row 1
        """
        freeze_cell = f'{freeze_col}{freeze_row}'
        ws.freeze_panes = freeze_cell
        logger.debug('Freeze panes set at: {}', freeze_cell)


def write_csv(comparison: list[dict[str, Any]], output_path: Path) -> None:
    """
    Write comparison results to CSV file.
    
    Creates a CSV file with correlation data. All columns from the comparison
    dictionaries are included in the output.
    
    Args:
        comparison: List of dictionaries containing correlation data.
                   All dictionaries must have the same keys.
        output_path: Path where the CSV file will be saved.
        
    Returns:
        None
        
    Side Effects:
        - Creates a CSV file at output_path
        - Logs progress and completion
        
    Raises:
        ValueError: If comparison list is empty
        IOError: If file cannot be written
        
    Example:
        >>> from pathlib import Path
        >>> from ttsi_corr.excel_writer import write_csv
        >>> 
        >>> data = [
        ...     {'Workload': 'BERT', 'Score': 100.0, 'Ratio': 0.95},
        ...     {'Workload': 'ResNet', 'Score': 200.0, 'Ratio': 1.05}
        ... ]
        >>> write_csv(data, Path('output/results.csv'))
        
    Note:
        Migrated from tools.run_ttsi_corr._write_csv() in Phase 2.
    """
    if not comparison:
        raise ValueError('Comparison list cannot be empty')
    
    logger.debug('Writing {} correlation results to: {}', len(comparison), output_path)
    
    with open(output_path, 'w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=comparison[0].keys())
        writer.writeheader()
        for row in comparison:
            writer.writerow(row)
    
    logger.info('CSV file written to: {}', output_path)

