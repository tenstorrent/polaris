# TTNN Workload Flow Documentation

This document provides a quick reference for the TTNN workload flow in Polaris.

## Quick Reference

### Parameter Flow

```
YAML Config         Polaris Dispatcher         TTNN Workload
───────────         ──────────────────         ─────────────
name: "Resnet50" ──► wln ──────────────────► wlname: str
path: "..." ───────► wpath (loaded as func)
cfg: {...} ────────► gcfg ──────────────────► cfg: dict
(device managed) ──► ttnn_device ───────────► device: TTNNDevice
```

### Code Locations

- **Dispatch Logic**: `polaris.py:111-126` (TTNN workload dispatch)
- **Reference Implementation**: `workloads/ttnn/resnet50/ttnn_functional_resnet50.py:1371-1402`
- **Configuration Examples**: `config/all_workloads.yaml`
- **User Guide**: `doc/user_guide.md` (see "TTNN Workload Development Guide" section)

### Standard Signature

```python
def run_<model_name>(wlname: str, device: TTNNDevice, cfg: dict):
    """
    Workload entry point for Polaris TTNN framework.
    
    Args:
        wlname: Workload identifier from Polaris (corresponds to 'wln' in 
               polaris.py:120). Example: "Resnet50"
               
        device: TTNN device instance (lifecycle managed by Polaris)
        
        cfg: Configuration dict with model hyperparameters
    
    Returns:
        Output tensor from model forward pass
        
    Note:
        Invoked by polaris.py:120 as: ttnn_func(wln, ttnn_device, gcfg)
        where wln comes from workload spec YAML (e.g., config/all_workloads.yaml)
    """
    pass
```

## Key Conventions

1. **Function Naming**: `run_<model_name>`
2. **Parameter Order**: `(wlname, device, cfg)` - always in this order
3. **Device Management**: Never open/close device - Polaris manages it
4. **Configuration Access**: Use `cfg.get('key', default)` with defaults
5. **Return Type**: TTNN Tensor

## Variable Name Mapping

| YAML Field | polaris.py Variable | Function Parameter | Purpose |
|------------|-------------------|-------------------|---------|
| `api: TTNN` | `wlg` | N/A | Workload API type |
| `name: "Resnet50"` | `wln` | `wlname` | Workload identifier |
| `instances.default` | `wli` | N/A | Instance name |
| `cfg: {batch_size: 8}` | `gcfg` | `cfg` | Configuration dict |
| (N/A - managed) | `ttnn_device` | `device` | Device instance |
| (N/A - runtime) | `wlb` | `cfg['batch_size']` | Batch size (if overridden) |

## Example Invocation Flow

1. **YAML Configuration** (`config/all_workloads.yaml`):
   ```yaml
   workloads:
     - api: TTNN
       name: "Resnet50"
       basedir: workloads/ttnn/resnet50
       module: run_resnet50@ttnn_functional_resnet50.py
       instances:
         default:
           batch_size: 8
           num_channels: 3
   ```

2. **Polaris Dispatch** (`polaris.py:111-126`):
   ```python
   # Load function from module path
   ttnn_func = get_ttnn_functional_instance(wpath, wln, gcfg)
   
   # Open device (managed by Polaris)
   ttnn_device = open_device(device_id=0)
   
   # Call with standard signature
   ttnn_res = ttnn_func(wln, ttnn_device, gcfg)
   #                     │         │         │
   #                     │         │         └─► cfg dict
   #                     │         └──────────► device instance
   #                     └────────────────────► wlname string
   
   # Close device (managed by Polaris)
   close_device(ttnn_device)
   ```

3. **Workload Implementation**:
   ```python
   def run_resnet50(wlname: str, device: TTNNDevice, cfg: dict):
       # Extract configuration with defaults
       batch_size = cfg.get('batch_size', 8)
       num_channels = cfg.get('num_channels', 3)
       
       # Create tensors using provided device
       input_tensor = ttnn.Tensor(
           shape=[batch_size, num_channels, 224, 224],
           dtype=ttnn.bfloat16,
           device=device  # Use provided device, don't create new one
       )
       
       # Implement model
       output = model(input_tensor, device, ...)
       
       return output
   ```

## Documentation Updates Made

### 1. Function Docstring
- **File**: `workloads/ttnn/resnet50/ttnn_functional_resnet50.py:1372-1392`
- **Content**: Added comprehensive docstring following Option 2 format
- **Details**: Documents parameter contract, polaris.py invocation, and usage

### 2. Dispatch Comments
- **File**: `polaris.py:116-119`
- **Content**: Added inline comments explaining parameter flow
- **Details**: Clarifies wln→wlname mapping and parameter purposes

### 3. User Guide Section
- **File**: `doc/user_guide.md`
- **Section**: "TTNN Workload Development Guide"
- **Content**: Comprehensive guide including:
  - Architecture overview
  - Standard signature with examples
  - YAML configuration format
  - Parameter flow diagram
  - Best practices
  - Debugging tips
  - Common issues

### 4. API Type Documentation
- **File**: `doc/user_guide.md:91-93`
- **Content**: Added supported workload APIs (TTSIM, TTNN, ONNX)

## For New Developers

### Creating a New TTNN Workload

1. **Create the workload file**:
   ```bash
   mkdir -p workloads/ttnn/my_model
   touch workloads/ttnn/my_model/ttnn_functional_my_model.py
   ```

2. **Implement with standard signature**:
   ```python
   def run_my_model(wlname: str, device: TTNNDevice, cfg: dict):
       # Extract config
       batch_size = cfg.get('batch_size', 1)
       
       # Implement model using device
       input_tensor = ttnn.Tensor(..., device=device)
       output = model(input_tensor)
       
       return output
   ```

3. **Add to YAML configuration**:
   ```yaml
   workloads:
     - api: TTNN
       name: MyModel
       basedir: workloads/ttnn/my_model
       module: run_my_model@ttnn_functional_my_model.py
       instances:
         default:
           batch_size: 8
   ```

4. **Run with Polaris**:
   ```bash
   python polaris.py \
       --wlspec config/my_workloads.yaml \
       --archspec config/tt_wh.yaml \
       --filterwl MyModel
   ```

## References

- **Main User Guide**: `doc/user_guide.md` - Complete TTNN workload guide
- **Reference Implementation**: `workloads/ttnn/resnet50/ttnn_functional_resnet50.py`
- **Polaris Dispatch**: `polaris.py:78-139` - `get_wlgraph()` function
- **TTNN API**: `ttsim/front/ttnn/` - TTNN tensor and operation implementations
- **Configuration Schema**: `tools/workloads.py` - Pydantic models for workload configs

## Support

For questions about TTNN workload development:
1. Review this documentation and the user guide
2. Check the reference implementation (ResNet-50)
3. Enable debug logging: `--log_level debug`
4. Consult the project repository: https://github.com/tenstorrent/polaris

