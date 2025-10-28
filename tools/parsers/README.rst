Markdown Table Parser
=====================

This directory contains the markdown table parser for extracting Tensix Metal performance metrics from markdown documents.

Parser Implementation
---------------------

The parser uses **markdown-it-py**, a robust, standards-compliant CommonMark parser for Python. This provides reliable parsing of markdown tables with proper handling of edge cases and format variations.

**Key Features:**

- ✅ CommonMark compliant parsing
- ✅ Token-based table extraction
- ✅ Robust handling of markdown variations
- ✅ Full type safety with mypy
- ✅ Built-in type hints (py.typed)
- ✅ Actively maintained library

Architecture
------------

The parser consists of:

1. **Pydantic Model** (``TensixMdPerfMetricModel``)
   
   - Type-safe data model for performance metrics
   - Supports multiple metric types (LLM, vision, detection, NLP, diffusion)
   - Optional fields for different model categories
   - Performance accessor methods

2. **Column Mapping** (``MD_COLNAME_MAP``)
   
   - Maps various column name formats to standard field names
   - Handles variations in naming conventions
   - Extensible for new column types

3. **Parser Functions**
   
   - ``extract_table_from_md_link()``: Main extraction function
   - ``save_md_metrics()``: Saves metrics to YAML files by type
   - Helper functions for cell cleaning and numeric parsing

Dependencies
------------

**Required:**

- ``markdown-it-py>=3.0.0`` - CommonMark parser (in environment.yaml)
- ``pydantic>=2.0`` - Data validation
- ``yaml`` - Output formatting
- ``loguru`` - Logging

Usage
-----

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from tools.parsers.md_parser import extract_table_from_md_link, save_md_metrics
   from pathlib import Path

   # Extract metrics from a markdown file
   url = 'https://raw.githubusercontent.com/tenstorrent/tt-metal/main/models/README.md'
   metrics = extract_table_from_md_link(url, use_cache=True)

   # Save to YAML files
   output_dir = Path('data/metal/inf')
   save_md_metrics(metrics, output_dir)

   print(f'Extracted {len(metrics)} metrics')

Command Line
^^^^^^^^^^^^

.. code-block:: bash

   # Parse TT-Metal models README
   python tools/parse_ttsi_perf_results.py \
       --input https://raw.githubusercontent.com/tenstorrent/tt-metal/main/models/README.md \
       --output-dir data/metal/inf

API Reference
-------------

extract_table_from_md_link()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def extract_table_from_md_link(link: str, use_cache: bool = True) -> List[TensixMdPerfMetricModel]:
       """
       Extracts tables from a Markdown link.
       
       Args:
           link (str): The URL of the MD page containing the tables.
           use_cache (bool): Whether to use cache for fetching content.
           
       Returns:
           List[TensixMdPerfMetricModel]: Extracted table data as Pydantic models.
           
       Raises:
           ValueError: If no valid tables found or data cannot be parsed.
       """

save_md_metrics()
^^^^^^^^^^^^^^^^^

.. code-block:: python

   def save_md_metrics(metrics: List[TensixMdPerfMetricModel], output_dir: Path) -> None:
       """
       Saves the extracted MD metrics to YAML files grouped by metric type.
       
       Args:
           metrics (List[TensixMdPerfMetricModel]): List of extracted metrics.
           output_dir (Path): Directory to save the metrics.
           
       Output:
           Creates separate YAML files for each metric type:
           - tensix_md_perf_metrics_llm.yaml
           - tensix_md_perf_metrics_vision.yaml
           - tensix_md_perf_metrics_detection.yaml
           - tensix_md_perf_metrics_nlp.yaml
           - tensix_md_perf_metrics_diffusion.yaml
       """

Output Format
-------------

The parser groups metrics by type and saves to separate YAML files:

.. code-block:: yaml

   # data/metal/inf/tensix_md_perf_metrics_llm.yaml
   - batch: 32
     hardware: n150 (Wormhole)
     model: Llama 3.1 8B
     release: v0.55.0
     tokens_per_sec_per_user: 23.5
     target_tokens_per_sec_per_user: 26.0
     ttft_ms: 89.0
     gpu: Tensix
     id: metal
     input_dtype: bf8
     precision: bf8

Migration Notes
---------------

This parser migrated from a regex-based implementation to markdown-it-py on 2025-10-21.

**Benefits of Migration:**

- 28% less code (286 vs 397 lines)
- 50% fewer regex operations
- Better standards compliance
- Easier to maintain and extend
- More robust error handling

**Breaking Changes:**

None - the API remains identical. The migration is transparent to users.

**Deprecated Functions:**

The following internal functions were removed (not part of public API):

- ``parse_md_table_row()`` - replaced by token-based parsing
- ``detect_content_format()`` - no longer needed
- ``clean_md_cell_value()`` - renamed to ``clean_cell_value()``

Troubleshooting
---------------

No tables found
^^^^^^^^^^^^^^^

**Issue**: "No valid table data extracted from URL"

**Solution**: 

- Verify the URL contains markdown tables
- Check that tables have a "model" column header
- Ensure tables follow standard markdown format

Parsing errors
^^^^^^^^^^^^^^

**Issue**: "Failed to create model from row"

**Solution**:

- Check that required fields (model, batch, hardware) are present
- Verify numeric values are properly formatted
- Review logs for specific validation errors

Future Enhancements
-------------------

Potential improvements for future versions:

1. Add explicit release table filtering for clarity
2. Support for additional metric types
3. Parallel processing for large files
4. Schema validation for input markdown
5. Better error recovery and reporting
