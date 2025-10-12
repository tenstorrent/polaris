Markdown Parser Usage Examples
===============================

Quick Start
-----------

Command Line Usage
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Parse TT-Metal models README and extract metrics
   python tools/parse_ttsi_perf_results.py \
       --input https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/models/README.md \
       --output-dir data/metal/inf

Expected output:

.. code-block:: text

   INFO: Parsing markdown tables from: https://...
   INFO: Extracted 150 metrics from MD file https://...
   INFO: Saved 85 MD metrics for type llm to data/metal/inf/tensix_md_perf_metrics_llm.yaml
   INFO: Saved 42 MD metrics for type vision to data/metal/inf/tensix_md_perf_metrics_vision.yaml
   INFO: Saved 23 MD metrics for type detection to data/metal/inf/tensix_md_perf_metrics_detection.yaml

Options
^^^^^^^

.. code-block:: bash

   # Disable caching (always fetch fresh data)
   python tools/parse_ttsi_perf_results.py \
       --input https://... \
       --output-dir data/metal/inf \
       --no-use-cache

   # Specify custom output directory
   python tools/parse_ttsi_perf_results.py \
       --input https://... \
       --output-dir /path/to/output

Programmatic Usage
------------------

Basic Example
^^^^^^^^^^^^^

.. code-block:: python

   from tools.parsers.md_parser import extract_table_from_md_link, save_md_metrics
   from pathlib import Path

   # Extract metrics
   url = 'https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/models/README.md'
   metrics = extract_table_from_md_link(url, use_cache=True)

   # Save to files
   output_dir = Path('data/metal/inf')
   save_md_metrics(metrics, output_dir)

   print(f'Extracted {len(metrics)} metrics')
   for metric in metrics[:3]:  # Show first 3
       print(f'  - {metric.model}: {metric.batch} batch on {metric.hardware}')

Processing Metrics
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from tools.parsers.md_parser import extract_table_from_md_link

   url = 'https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/models/README.md'
   metrics = extract_table_from_md_link(url)

   # Filter by hardware
   n150_metrics = [m for m in metrics if 'n150' in m.hardware.lower()]
   print(f'Found {len(n150_metrics)} metrics for n150')

   # Filter by model type
   llm_metrics = [m for m in metrics if m.tokens_per_sec_per_user is not None]
   print(f'Found {len(llm_metrics)} LLM metrics')

   # Calculate average performance
   avg_perf = sum(m.get_perf() for m in llm_metrics if m.get_perf()) / len(llm_metrics)
   print(f'Average LLM performance: {avg_perf:.2f} tokens/sec/user')

Custom Processing
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from tools.parsers.md_parser import extract_table_from_md_link
   import pandas as pd

   # Extract metrics
   url = 'https://...'
   metrics = extract_table_from_md_link(url)

   # Convert to DataFrame for analysis
   data = [m.model_dump() for m in metrics]
   df = pd.DataFrame(data)

   # Analyze by hardware
   grouped = df.groupby('hardware')['tokens_per_sec_per_user'].mean()
   print(grouped)

Working with Pydantic Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from tools.parsers.md_parser import TensixMdPerfMetricModel

   # Access typed attributes
   metric = metrics[0]
   print(f'Model: {metric.model}')
   print(f'Batch size: {metric.batch}')
   print(f'Hardware: {metric.hardware}')

   # Use accessor methods
   perf = metric.get_perf()  # Gets appropriate perf metric
   target = metric.get_target_perf()  # Gets target perf metric
   metric_name = metric.get_metric_name()  # Gets metric name

   print(f'{metric_name}: {perf} (target: {target})')

   # Export to dict or JSON
   as_dict = metric.model_dump()
   as_json = metric.model_dump_json()

Output Format
-------------

YAML Structure
^^^^^^^^^^^^^^

Metrics are saved to separate YAML files by type:

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

   - batch: 32
     hardware: n300 (Wormhole)
     model: Llama 3.2 1B
     ...

File Organization
^^^^^^^^^^^^^^^^^

+---------------------------------------+------------------------+
| File                                  | Content                |
+=======================================+========================+
| tensix_md_perf_metrics_llm.yaml      | LLM models             |
+---------------------------------------+------------------------+
| tensix_md_perf_metrics_vision.yaml   | Vision models          |
+---------------------------------------+------------------------+
| tensix_md_perf_metrics_detection.yaml| Object detection models|
+---------------------------------------+------------------------+
| tensix_md_perf_metrics_nlp.yaml      | NLP models             |
+---------------------------------------+------------------------+
| tensix_md_perf_metrics_diffusion.yaml| Diffusion models       |
+---------------------------------------+------------------------+

Integration Examples
--------------------

With Correlation Tool
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from tools.parsers.md_parser import extract_table_from_md_link
   from tools.run_ttsi_corr import normalize_md_metric

   # Extract and normalize metrics for correlation
   url = 'https://...'
   md_metrics = extract_table_from_md_link(url)

   for md_metric in md_metrics:
       normalized = normalize_md_metric(md_metric)
       if normalized:
           # Use in correlation analysis
           process_metric(normalized)

Custom Validator
^^^^^^^^^^^^^^^^

.. code-block:: python

   from tools.parsers.md_parser import TensixMdPerfMetricModel
   from pydantic import validator

   class CustomMetricModel(TensixMdPerfMetricModel):
       @validator('batch')
       def validate_batch(cls, v):
           if v <= 0:
               raise ValueError('Batch size must be positive')
           return v

       @validator('tokens_per_sec_per_user')
       def validate_performance(cls, v):
           if v is not None and v < 0:
               raise ValueError('Performance cannot be negative')
           return v

Troubleshooting
---------------

Issue: "ModuleNotFoundError: No module named 'markdown_it'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Solution**: Install the dependency:

.. code-block:: bash

   conda install markdown-it-py=3.0.0

Or update your environment:

.. code-block:: bash

   conda env update -f environment.yaml

Issue: "No valid table data extracted"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Possible causes:**

1. URL doesn't contain markdown tables
2. Tables don't have required "model" column
3. Network/access issues

**Solution**:

.. code-block:: python

   # Add error handling
   try:
       metrics = extract_table_from_md_link(url)
   except ValueError as e:
       print(f'Parsing failed: {e}')
       # Check URL manually or verify table format

Issue: "Failed to create model from row"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause**: Missing required fields or invalid data types

**Solution**: Enable debug logging to see which rows fail:

.. code-block:: python

   from loguru import logger
   
   logger.remove()
   logger.add(sys.stdout, level='DEBUG')
   
   metrics = extract_table_from_md_link(url)

Issue: Different results on subsequent runs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause**: Caching behavior

**Solution**: Disable cache for testing:

.. code-block:: python

   metrics = extract_table_from_md_link(url, use_cache=False)

Performance Tips
----------------

Caching
^^^^^^^

The parser caches URL content by default. For production:

.. code-block:: python

   # Development: Fresh data every time
   metrics = extract_table_from_md_link(url, use_cache=False)

   # Production: Use cache for performance
   metrics = extract_table_from_md_link(url, use_cache=True)

Batch Processing
^^^^^^^^^^^^^^^^

Process multiple URLs efficiently:

.. code-block:: python

   urls = [
       'https://url1.md',
       'https://url2.md',
       'https://url3.md',
   ]

   all_metrics = []
   for url in urls:
       try:
           metrics = extract_table_from_md_link(url)
           all_metrics.extend(metrics)
       except Exception as e:
           print(f'Failed to parse {url}: {e}')

   print(f'Total metrics: {len(all_metrics)}')

Best Practices
--------------

1. **Error Handling**: Always wrap parsing in try-except blocks
2. **Validation**: Verify metrics have expected fields before using
3. **Logging**: Use debug level logging during development
4. **Caching**: Use cache in production, disable for testing
5. **Type Checking**: Leverage Pydantic models for type safety

Advanced Usage
--------------

Custom Column Mapping
^^^^^^^^^^^^^^^^^^^^^

Extend column mapping for new formats:

.. code-block:: python

   from tools.parsers.md_parser import MD_COLNAME_MAP

   # Add custom mappings
   MD_COLNAME_MAP['custom_metric'] = 'my_field'

   # Then parse as usual
   metrics = extract_table_from_md_link(url)

Filtering During Parse
^^^^^^^^^^^^^^^^^^^^^^

Post-process to filter specific models:

.. code-block:: python

   url = 'https://...'
   all_metrics = extract_table_from_md_link(url)

   # Filter for specific models
   target_models = ['Llama 3.1 8B', 'Llama 3.2 1B']
   filtered = [m for m in all_metrics if m.model in target_models]

Migration Notes
---------------

If you previously used the regex-based parser, the API is identical:

.. code-block:: python

   # Old code (still works)
   from tools.parsers.md_parser import extract_table_from_md_link
   metrics = extract_table_from_md_link(url)

   # No changes needed!

**Removed Functions** (internal only, not part of public API):

- ``parse_md_table_row()``
- ``detect_content_format()``
- ``clean_md_cell_value()`` (renamed to ``clean_cell_value()``)
