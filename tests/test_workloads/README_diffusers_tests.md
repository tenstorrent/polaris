# Diffusers Module Test Suite

## Overview

Comprehensive unit tests for `workloads/diffusers` modules using pytest framework.

## Test Files

1. **`test_diffusers.py`** - Full pytest test suite with fixtures and parametrized tests
2. **`test_diffusers_standalone.py`** - Standalone test runner (no pytest dependency)

## Running Tests

### With pytest (Recommended)

```bash
# Run all diffusers tests
pytest tests/test_workloads/test_diffusers.py -v

# Run specific test class
pytest tests/test_workloads/test_diffusers.py::TestDownsample2D -v

# Run with coverage
pytest tests/test_workloads/test_diffusers.py --cov=workloads/diffusers --cov-report=html

# Run specific test
pytest tests/test_workloads/test_diffusers.py::TestDownsample2D::test_non_conv_downsampling_not_implemented -v
```

### Without pytest (Standalone)

```bash
python tests/test_workloads/test_diffusers_standalone.py
```

## Test Coverage

### Downsampling Module (`downsampling.py`)
- ✅ Initialization with convolution
- ✅ Non-convolutional downsampling error (NotImplementedError)
- ✅ Channel mismatch validation (ValueError)
- ✅ RMSNorm not supported error
- ✅ Forward pass with normalization error
- ✅ Input channel validation in forward pass

### Upsampling Module (`upsampling.py`)
- ✅ Basic initialization
- ✅ LayerNorm not supported error
- ✅ RMSNorm not supported error
- ✅ ConvTranspose2d not supported error
- ✅ Input channel validation in forward pass

### ResNet Module (`resnet.py`)
- ✅ Basic initialization
- ✅ FIR upsampling not supported error
- ✅ SDE_VP upsampling not supported error
- ✅ FIR downsampling not supported error
- ✅ SDE_VP downsampling not supported error
- ✅ Scale-shift time embedding error

### Transformer Module (`attention.py`)
- ✅ Gated attention not supported error
- ✅ Ada_norm_single not supported error
- ✅ Chunked feed forward not supported error

### Attention Processor Module (`attention_processor.py`)
- ✅ Basic initialization
- ✅ Spatial norm not supported error
- ✅ QK norm not supported error
- ✅ 4D input not supported error
- ✅ Attention mask not supported error
- ✅ Group norm not supported error

### Integration Tests
- ✅ Module imports verification
- ✅ Stable diffusion test file existence

## Test Statistics

- **Total Test Cases**: 27+
- **Test Classes**: 6
- **Modules Covered**: 5 core modules
- **Error Types Tested**: NotImplementedError, ValueError

## Error Handling Verification

All tests verify that:
1. **Proper exception types** are raised (NotImplementedError for unimplemented features, ValueError for validation)
2. **Error messages are specific** and include parameter names/values
3. **Error messages are actionable** for developers debugging issues

## Dependencies

Required for running tests:
- pytest >= 6.0
- numpy
- ttsim modules

Optional:
- pytest-cov (for coverage reports)
- pytest-xdist (for parallel test execution)

## Adding New Tests

### Template for New Test

```python
class TestNewModule:
    """Test NewModule module."""
    
    def test_feature_not_implemented(self):
        """Test that unimplemented feature raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            NewModule(
                objname="test",
                unsupported_param=True
            )
        
        assert "specific feature name" in str(exc_info.value)
        assert "parameter='value'" in str(exc_info.value)
```

## CI/CD Integration

Add to CI pipeline:
```yaml
- name: Run Diffusers Tests
  run: |
    pytest tests/test_workloads/test_diffusers.py \
      --junitxml=test-results/diffusers.xml \
      --cov=workloads/diffusers \
      --cov-report=xml
```

## Known Issues / Limitations

1. **Environment Dependencies**: Tests require full ttsim environment with all dependencies installed
2. **Module Imports**: Some tests may fail if onnx or other optional dependencies are missing
3. **Forward Pass Tests**: Limited forward pass testing due to complexity of creating valid tensor inputs

## Future Improvements

- [ ] Add forward pass integration tests
- [ ] Add tests for successful execution paths (not just error cases)
- [ ] Add property-based testing with hypothesis
- [ ] Add performance/benchmark tests
- [ ] Mock external dependencies to reduce test dependencies
- [ ] Add tests for UNet2D and VAE modules

## Related Documentation

- [Main README](../../README.md)
- [Diffusers Documentation](../../doc/README.md)
- [Testing Guide](../../doc/tools/ci/testing.md)

