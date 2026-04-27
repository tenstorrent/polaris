#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Test file for MSDeformAttn module: Shape inference and numerical validation.
Compares TTSim implementation against PyTorch reference module-by-module.
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

# Add polaris root to path
# Locate polaris root (contains pyproject.toml) regardless of run directory
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.exists(
    os.path.join(_root, "pyproject.toml")
):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from ttsim.ops.tensor import SimTensor
from workloads.Deformable_DETR.models.ops.modules.ms_deform_attn_ttsim import (
    MSDeformAttn as MSDeformAttnTTSim,
)
from workloads.Deformable_DETR.reference.ms_deform_attn import MSDeformAttn


def compute_error_metrics(pytorch_val, ttsim_val):
    """Compute error metrics between PyTorch and TTSim outputs."""
    abs_diff = np.abs(pytorch_val - ttsim_val)
    max_abs_error = np.max(abs_diff)
    mean_abs_error = np.mean(abs_diff)

    pytorch_norm = np.linalg.norm(pytorch_val.flatten())
    ttsim_norm = np.linalg.norm(ttsim_val.flatten())
    rel_error = np.linalg.norm(abs_diff.flatten()) / (pytorch_norm + 1e-8)

    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "relative_error": rel_error,
        "pytorch_norm": pytorch_norm,
        "ttsim_norm": ttsim_norm,
    }


def format_array_sample(arr, max_elements=10):
    """Format array sample for display."""
    flat = arr.flatten()
    if len(flat) <= max_elements:
        return f"[{', '.join(f'{x:.6f}' for x in flat)}]"
    else:
        first_part = ", ".join(f"{x:.6f}" for x in flat[: max_elements // 2])
        last_part = ", ".join(f"{x:.6f}" for x in flat[-max_elements // 2 :])
        return f"[{first_part}, ..., {last_part}]"


def test_shape_inference():
    """Test shape inference mode (data=None)."""
    print("\n" + "=" * 80)
    print("SHAPE INFERENCE TEST - MSDeformAttn Module")
    print("=" * 80)

    # Configuration
    d_model, n_levels, n_heads, n_points = 256, 4, 8, 4
    N, Len_q = 2, 100
    spatial_shapes = np.array([[50, 50], [25, 25], [13, 13], [7, 7]], dtype=np.int32)
    Len_in = int(np.sum(spatial_shapes[:, 0] * spatial_shapes[:, 1]))

    print(
        f"Config: d_model={d_model}, n_levels={n_levels}, n_heads={n_heads}, n_points={n_points}"
    )
    print(f"Input: N={N}, Len_q={Len_q}, Len_in={Len_in}")

    # Create TTSim module
    module_ttsim = MSDeformAttnTTSim(
        d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points
    )

    # Create shape-only inputs
    query_sim = SimTensor(
        {
            "name": "query",
            "shape": [N, Len_q, d_model],
            "data": None,
            "dtype": np.dtype("float32"),
        }
    )
    reference_points_sim = SimTensor(
        {
            "name": "reference_points",
            "shape": [N, Len_q, n_levels, 2],
            "data": None,
            "dtype": np.dtype("float32"),
        }
    )
    input_flatten_sim = SimTensor(
        {
            "name": "input_flatten",
            "shape": [N, Len_in, d_model],
            "data": None,
            "dtype": np.dtype("float32"),
        }
    )
    input_spatial_shapes_sim = SimTensor(
        {
            "name": "input_spatial_shapes",
            "shape": [n_levels, 2],
            "data": None,
            "dtype": np.dtype("int32"),
        }
    )

    # Forward pass
    output_sim = module_ttsim.forward(
        query_sim, reference_points_sim, input_flatten_sim, input_spatial_shapes_sim
    )

    expected_shape = [N, Len_q, d_model]
    actual_shape = output_sim.shape

    result = {
        "test": "Shape Inference",
        "config": f"d_model={d_model}, n_levels={n_levels}, n_heads={n_heads}, n_points={n_points}, N={N}, Len_q={Len_q}",
        "expected_shape": expected_shape,
        "actual_shape": actual_shape,
        "passed": expected_shape == actual_shape,
    }

    print(f"\nExpected output shape: {expected_shape}")
    print(f"Actual output shape: {actual_shape}")
    print(f"Test: {'PASSED' if result['passed'] else 'FAILED'}")

    return result


def pytorch_forward_with_intermediates(
    module_pytorch, query, reference_points, input_flatten, input_spatial_shapes
):
    """PyTorch forward with intermediate outputs."""
    N, Len_q, _ = query.shape
    N_in, Len_in, _ = input_flatten.shape

    intermediates = {}

    # Value projection
    value = module_pytorch.value_proj(input_flatten)
    intermediates["value_proj_shape"] = list(value.shape)
    intermediates["value_proj"] = value.detach().cpu().numpy()

    # Reshape value
    value = value.view(
        N,
        Len_in,
        module_pytorch.n_heads,
        module_pytorch.d_model // module_pytorch.n_heads,
    )
    intermediates["value_reshaped_shape"] = list(value.shape)
    intermediates["value_reshaped"] = value.detach().cpu().numpy()

    # Sampling offsets
    sampling_offsets = module_pytorch.sampling_offsets(query)
    intermediates["sampling_offsets_linear_shape"] = list(sampling_offsets.shape)
    intermediates["sampling_offsets_linear"] = sampling_offsets.detach().cpu().numpy()

    sampling_offsets = sampling_offsets.view(
        N,
        Len_q,
        module_pytorch.n_heads,
        module_pytorch.n_levels,
        module_pytorch.n_points,
        2,
    )
    intermediates["sampling_offsets_reshaped_shape"] = list(sampling_offsets.shape)
    intermediates["sampling_offsets_reshaped"] = sampling_offsets.detach().cpu().numpy()

    # Attention weights
    attention_weights = module_pytorch.attention_weights(query)
    intermediates["attention_weights_linear_shape"] = list(attention_weights.shape)
    intermediates["attention_weights_linear"] = attention_weights.detach().cpu().numpy()

    attention_weights = attention_weights.view(
        N,
        Len_q,
        module_pytorch.n_heads,
        module_pytorch.n_levels * module_pytorch.n_points,
    )
    intermediates["attention_weights_reshaped_shape"] = list(attention_weights.shape)

    attention_weights = F.softmax(attention_weights, -1)
    intermediates["attention_weights_softmax_shape"] = list(attention_weights.shape)
    intermediates["attention_weights_softmax"] = (
        attention_weights.detach().cpu().numpy()
    )

    attention_weights = attention_weights.view(
        N,
        Len_q,
        module_pytorch.n_heads,
        module_pytorch.n_levels,
        module_pytorch.n_points,
    )
    intermediates["attention_weights_final_shape"] = list(attention_weights.shape)
    intermediates["attention_weights_final"] = attention_weights.detach().cpu().numpy()

    # Sampling locations
    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
    elif reference_points.shape[-1] == 4:
        sampling_locations = (
            reference_points[:, :, None, :, None, :2]
            + sampling_offsets
            / module_pytorch.n_points
            * reference_points[:, :, None, :, None, 2:]
            * 0.5
        )

    intermediates["sampling_locations_shape"] = list(sampling_locations.shape)
    intermediates["sampling_locations"] = sampling_locations.detach().cpu().numpy()

    # MS Deform Attn Core
    from workloads.Deformable_DETR.reference.ms_deform_attn_func import (
        ms_deform_attn_core_pytorch,
    )

    ms_output = ms_deform_attn_core_pytorch(
        value, input_spatial_shapes, sampling_locations, attention_weights
    )
    intermediates["ms_deform_attn_output_shape"] = list(ms_output.shape)
    intermediates["ms_deform_attn_output"] = ms_output.detach().cpu().numpy()

    # Output projection
    output = module_pytorch.output_proj(ms_output)
    intermediates["output_shape"] = list(output.shape)
    intermediates["output"] = output.detach().cpu().numpy()

    return output, intermediates


def ttsim_forward_with_intermediates(
    module_ttsim, query, reference_points, input_flatten, input_spatial_shapes
):
    """TTSim forward with intermediate outputs."""
    N, Len_q, _ = query.shape
    N_in, Len_in, _ = input_flatten.shape

    intermediates = {}

    # Convert to SimTensor
    query_sim = SimTensor(
        {
            "name": "query",
            "shape": list(query.shape),
            "data": query,
            "dtype": query.dtype,
        }
    )
    reference_points_sim = SimTensor(
        {
            "name": "reference_points",
            "shape": list(reference_points.shape),
            "data": reference_points,
            "dtype": reference_points.dtype,
        }
    )
    input_flatten_sim = SimTensor(
        {
            "name": "input_flatten",
            "shape": list(input_flatten.shape),
            "data": input_flatten,
            "dtype": input_flatten.dtype,
        }
    )
    input_spatial_shapes_sim = SimTensor(
        {
            "name": "input_spatial_shapes",
            "shape": list(input_spatial_shapes.shape),
            "data": input_spatial_shapes,
            "dtype": input_spatial_shapes.dtype,
        }
    )

    # Value projection
    value = module_ttsim.value_proj(input_flatten_sim)
    intermediates["value_proj_shape"] = value.shape
    intermediates["value_proj"] = value.data

    # Reshape value
    value.shape = [
        N,
        Len_in,
        module_ttsim.n_heads,
        module_ttsim.d_model // module_ttsim.n_heads,
    ]
    value.data = value.data.reshape(value.shape)
    intermediates["value_reshaped_shape"] = value.shape
    intermediates["value_reshaped"] = value.data

    # Sampling offsets
    sampling_offsets = module_ttsim.sampling_offsets(query_sim)
    intermediates["sampling_offsets_linear_shape"] = sampling_offsets.shape
    intermediates["sampling_offsets_linear"] = sampling_offsets.data

    sampling_offsets.shape = [
        N,
        Len_q,
        module_ttsim.n_heads,
        module_ttsim.n_levels,
        module_ttsim.n_points,
        2,
    ]
    sampling_offsets.data = sampling_offsets.data.reshape(sampling_offsets.shape)
    intermediates["sampling_offsets_reshaped_shape"] = sampling_offsets.shape
    intermediates["sampling_offsets_reshaped"] = sampling_offsets.data

    # Attention weights
    attention_weights = module_ttsim.attention_weights(query_sim)
    intermediates["attention_weights_linear_shape"] = attention_weights.shape
    intermediates["attention_weights_linear"] = attention_weights.data

    attention_weights.shape = [
        N,
        Len_q,
        module_ttsim.n_heads,
        module_ttsim.n_levels * module_ttsim.n_points,
    ]
    attention_weights.data = attention_weights.data.reshape(attention_weights.shape)
    intermediates["attention_weights_reshaped_shape"] = attention_weights.shape

    # Softmax
    from types import SimpleNamespace
    from ttsim.ops.desc.helpers import unary_fwd

    attention_weights_softmax = SimTensor(
        {
            "name": "attention_weights_softmax",
            "shape": None,
            "data": None,
            "dtype": None,
        }
    )
    op_softmax = SimpleNamespace(
        attrs={"axis": -1}, optype="Softmax", name="softmax", precision="fp32"
    )
    unary_fwd([attention_weights], [attention_weights_softmax], op_softmax)
    attention_weights = attention_weights_softmax

    intermediates["attention_weights_softmax_shape"] = attention_weights.shape
    intermediates["attention_weights_softmax"] = attention_weights.data

    attention_weights.shape = [
        N,
        Len_q,
        module_ttsim.n_heads,
        module_ttsim.n_levels,
        module_ttsim.n_points,
    ]
    attention_weights.data = attention_weights.data.reshape(attention_weights.shape)
    intermediates["attention_weights_final_shape"] = attention_weights.shape
    intermediates["attention_weights_final"] = attention_weights.data

    # Sampling locations
    sampling_locations = module_ttsim._compute_sampling_locations(
        reference_points_sim, sampling_offsets, input_spatial_shapes_sim
    )
    intermediates["sampling_locations_shape"] = sampling_locations.shape
    intermediates["sampling_locations"] = sampling_locations.data

    # MS Deform Attn Core
    from workloads.Deformable_DETR.models.ops.functions.ms_deform_attn_func_ttsim import (
        ms_deform_attn_core_ttsim,
    )

    ms_output = ms_deform_attn_core_ttsim(
        value, input_spatial_shapes_sim, sampling_locations, attention_weights
    )
    intermediates["ms_deform_attn_output_shape"] = ms_output.shape
    intermediates["ms_deform_attn_output"] = ms_output.data

    # Output projection
    output = module_ttsim.output_proj(ms_output)
    intermediates["output_shape"] = output.shape
    intermediates["output"] = output.data

    return output, intermediates


def _run_numerical_computation(
    config_name,
    d_model,
    n_levels,
    n_heads,
    n_points,
    N,
    Len_q,
    spatial_shapes,
    ref_points_dim=2,
):
    """Test numerical computation with detailed intermediate comparisons."""
    print(f"\n{'='*80}")
    print(f"NUMERICAL TEST - {config_name}")
    print(f"{'='*80}")
    print(
        f"Config: d_model={d_model}, n_levels={n_levels}, n_heads={n_heads}, n_points={n_points}"
    )
    print(f"Input: N={N}, Len_q={Len_q}, ref_points_dim={ref_points_dim}")
    print(f"Spatial shapes: {spatial_shapes.tolist()}")

    Len_in = int(np.sum(spatial_shapes[:, 0] * spatial_shapes[:, 1]))

    # Generate test data
    np.random.seed(42)
    torch.manual_seed(42)

    query_np = np.random.randn(N, Len_q, d_model).astype(np.float32)
    reference_points_np = np.random.rand(N, Len_q, n_levels, ref_points_dim).astype(
        np.float32
    )
    input_flatten_np = np.random.randn(N, Len_in, d_model).astype(np.float32)

    # PyTorch module
    module_pytorch = MSDeformAttn(
        d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points
    )
    module_pytorch.eval()

    # TTSim module with same weights
    module_ttsim = MSDeformAttnTTSim(
        d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points
    )

    # Copy weights from PyTorch to TTSim (SimNN.Linear uses .param.data / .bias.data)
    module_ttsim.sampling_offsets.param.data = (
        module_pytorch.sampling_offsets.weight.data.cpu().numpy()
    )
    module_ttsim.sampling_offsets.bias.data = (
        module_pytorch.sampling_offsets.bias.data.cpu().numpy()
    )
    module_ttsim.attention_weights.param.data = (
        module_pytorch.attention_weights.weight.data.cpu().numpy()
    )
    module_ttsim.attention_weights.bias.data = (
        module_pytorch.attention_weights.bias.data.cpu().numpy()
    )
    module_ttsim.value_proj.param.data = (
        module_pytorch.value_proj.weight.data.cpu().numpy()
    )
    module_ttsim.value_proj.bias.data = (
        module_pytorch.value_proj.bias.data.cpu().numpy()
    )
    module_ttsim.output_proj.param.data = (
        module_pytorch.output_proj.weight.data.cpu().numpy()
    )
    module_ttsim.output_proj.bias.data = (
        module_pytorch.output_proj.bias.data.cpu().numpy()
    )

    # PyTorch computation
    query_torch = torch.from_numpy(query_np)
    reference_points_torch = torch.from_numpy(reference_points_np)
    input_flatten_torch = torch.from_numpy(input_flatten_np)
    spatial_shapes_torch = torch.from_numpy(spatial_shapes)

    with torch.no_grad():
        output_torch, pytorch_inter = pytorch_forward_with_intermediates(
            module_pytorch,
            query_torch,
            reference_points_torch,
            input_flatten_torch,
            spatial_shapes_torch,
        )

    # TTSim computation - pass numpy array
    output_ttsim, ttsim_inter = ttsim_forward_with_intermediates(
        module_ttsim, query_np, reference_points_np, input_flatten_np, spatial_shapes
    )

    # Compare intermediates
    results = []
    tolerance = {"max_abs": 1e-4, "mean_abs": 1e-5, "relative": 1e-3}

    comparisons = [
        ("value_proj", "Value Projection"),
        ("value_reshaped", "Value Reshaped"),
        ("sampling_offsets_linear", "Sampling Offsets Linear"),
        ("sampling_offsets_reshaped", "Sampling Offsets Reshaped"),
        ("attention_weights_linear", "Attention Weights Linear"),
        ("attention_weights_softmax", "Attention Weights Softmax"),
        ("attention_weights_final", "Attention Weights Final"),
        ("sampling_locations", "Sampling Locations"),
        ("ms_deform_attn_output", "MS Deform Attn Output"),
        ("output", "Final Output"),
    ]

    for key, description in comparisons:
        if key in pytorch_inter and key in ttsim_inter:
            pytorch_val = pytorch_inter[key]
            ttsim_val = ttsim_inter[key]

            metrics = compute_error_metrics(pytorch_val, ttsim_val)

            passed = (
                metrics["max_abs_error"] < tolerance["max_abs"]
                and metrics["mean_abs_error"] < tolerance["mean_abs"]
                and metrics["relative_error"] < tolerance["relative"]
            )

            # Sample values
            pytorch_sample = format_array_sample(pytorch_val.flatten()[:10])
            ttsim_sample = format_array_sample(ttsim_val.flatten()[:10])

            results.append(
                {
                    "module": description,
                    "shape": list(pytorch_val.shape),
                    "metrics": metrics,
                    "pytorch_sample": pytorch_sample,
                    "ttsim_sample": ttsim_sample,
                    "passed": passed,
                }
            )

            print(f"\n{description}:")
            print(f"  Shape: {list(pytorch_val.shape)}")
            print(f"  PyTorch sample: {pytorch_sample}")
            print(f"  TTSim sample: {ttsim_sample}")
            print(
                f"  Max abs error: {metrics['max_abs_error']:.6e} (threshold: {tolerance['max_abs']:.6e})"
            )
            print(
                f"  Mean abs error: {metrics['mean_abs_error']:.6e} (threshold: {tolerance['mean_abs']:.6e})"
            )
            print(
                f"  Relative error: {metrics['relative_error']:.6e} (threshold: {tolerance['relative']:.6e})"
            )
            print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return {
        "config_name": config_name,
        "config": {
            "d_model": d_model,
            "n_levels": n_levels,
            "n_heads": n_heads,
            "n_points": n_points,
            "N": N,
            "Len_q": Len_q,
            "ref_points_dim": ref_points_dim,
            "spatial_shapes": spatial_shapes.tolist(),
        },
        "tolerance": tolerance,
        "results": results,
        "overall_passed": all(r["passed"] for r in results),
    }


def generate_markdown_report(shape_result, numerical_results, output_path):
    """Generate markdown report."""
    md = []
    md.append(f"# MSDeformAttn Module - Test Report")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"\n---\n")

    # Shape inference
    md.append(f"## Shape Inference Test")
    md.append(f"**Configuration:** {shape_result['config']}")
    md.append(f"- Expected shape: `{shape_result['expected_shape']}`")
    md.append(f"- Actual shape: `{shape_result['actual_shape']}`")
    status = "PASSED" if shape_result["passed"] else "FAILED"
    md.append(f"- **Result: {status}**")

    # Numerical tests
    md.append(f"\n---\n")
    md.append(f"## Numerical Computation Tests")

    for test in numerical_results:
        md.append(f"\n### {test['config_name']}")
        md.append(f"**Configuration:**")
        md.append(f"```")
        for k, v in test["config"].items():
            md.append(f"{k}: {v}")
        md.append(f"```")

        md.append(f"\n**Tolerance Thresholds:**")
        md.append(f"- Max absolute error: `{test['tolerance']['max_abs']:.6e}`")
        md.append(f"- Mean absolute error: `{test['tolerance']['mean_abs']:.6e}`")
        md.append(f"- Relative error: `{test['tolerance']['relative']:.6e}`")

        md.append(f"\n**Module-by-Module Results:**")
        md.append(
            f"\n| Module | Shape | Max Abs Error | Mean Abs Error | Rel Error | Status |"
        )
        md.append(
            f"|--------|-------|---------------|----------------|-----------|--------|"
        )

        for r in test["results"]:
            status = "PASS" if r["passed"] else "FAIL"
            shape_str = str(r["shape"])
            md.append(
                f"| {r['module']} | `{shape_str}` | "
                f"`{r['metrics']['max_abs_error']:.6e}` | "
                f"`{r['metrics']['mean_abs_error']:.6e}` | "
                f"`{r['metrics']['relative_error']:.6e}` | "
                f"{status} |"
            )

        md.append(f"\n**Sample Output Values:**")
        for r in test["results"]:
            md.append(f"\n*{r['module']}:*")
            md.append(f"- PyTorch: `{r['pytorch_sample']}`")
            md.append(f"- TTSim: `{r['ttsim_sample']}`")

        overall = "PASSED" if test["overall_passed"] else "FAILED"
        md.append(f"\n**Overall Result: {overall}**")

    # Summary
    md.append(f"\n---\n")
    md.append(f"## Summary")
    shape_status = "PASSED" if shape_result["passed"] else "FAILED"
    md.append(f"- Shape inference: {shape_status}")

    total_tests = len(numerical_results)
    passed_tests = sum(1 for t in numerical_results if t["overall_passed"])
    md.append(f"- Numerical tests: {passed_tests}/{total_tests} passed")

    all_passed = shape_result["passed"] and passed_tests == total_tests
    final_status = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    md.append(f"\n**Final Result: {final_status}**")

    # Write report
    report_content = "\n".join(md)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n{'='*80}")
    print(f"Report saved to: {output_path}")
    print(f"{'='*80}")

    return report_content


def main():
    """Run all tests and generate report."""
    # Shape inference test
    shape_result = test_shape_inference()

    # Numerical computation tests
    numerical_results = []

    # Test 1: Standard config with 2D reference points
    numerical_results.append(
        _run_numerical_computation(
            "Standard Configuration (2D Reference Points)",
            d_model=256,
            n_levels=4,
            n_heads=8,
            n_points=4,
            N=2,
            Len_q=100,
            spatial_shapes=np.array(
                [[50, 50], [25, 25], [13, 13], [7, 7]], dtype=np.int32
            ),
            ref_points_dim=2,
        )
    )

    # Test 2: Standard config with 4D reference points
    numerical_results.append(
        _run_numerical_computation(
            "Standard Configuration (4D Reference Points)",
            d_model=256,
            n_levels=4,
            n_heads=8,
            n_points=4,
            N=2,
            Len_q=100,
            spatial_shapes=np.array(
                [[50, 50], [25, 25], [13, 13], [7, 7]], dtype=np.int32
            ),
            ref_points_dim=4,
        )
    )

    # Test 3: Small config
    numerical_results.append(
        _run_numerical_computation(
            "Small Configuration",
            d_model=128,
            n_levels=2,
            n_heads=4,
            n_points=2,
            N=1,
            Len_q=50,
            spatial_shapes=np.array([[32, 32], [16, 16]], dtype=np.int32),
            ref_points_dim=2,
        )
    )

    # Generate report
    reports_dir = os.path.join(_root, "workloads", "Deformable_DETR", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    output_path = os.path.join(reports_dir, "ms_deform_attn_validation.md")

    generate_markdown_report(shape_result, numerical_results, output_path)


def test_numerical_computation_standard_2d():
    """Pytest wrapper: Standard config with 2D reference points."""
    result = _run_numerical_computation(
        "Standard Configuration (2D Reference Points)",
        d_model=256,
        n_levels=4,
        n_heads=8,
        n_points=4,
        N=2,
        Len_q=100,
        spatial_shapes=np.array([[50, 50], [25, 25], [13, 13], [7, 7]], dtype=np.int32),
        ref_points_dim=2,
    )
    assert result["overall_passed"], f"Numerical test failed: {result['config_name']}"


def test_numerical_computation_standard_4d():
    """Pytest wrapper: Standard config with 4D reference points."""
    result = _run_numerical_computation(
        "Standard Configuration (4D Reference Points)",
        d_model=256,
        n_levels=4,
        n_heads=8,
        n_points=4,
        N=2,
        Len_q=100,
        spatial_shapes=np.array([[50, 50], [25, 25], [13, 13], [7, 7]], dtype=np.int32),
        ref_points_dim=4,
    )
    assert result["overall_passed"], f"Numerical test failed: {result['config_name']}"


def test_numerical_computation_small():
    """Pytest wrapper: Small configuration."""
    result = _run_numerical_computation(
        "Small Configuration",
        d_model=128,
        n_levels=2,
        n_heads=4,
        n_points=2,
        N=1,
        Len_q=50,
        spatial_shapes=np.array([[32, 32], [16, 16]], dtype=np.int32),
        ref_points_dim=2,
    )
    assert result["overall_passed"], f"Numerical test failed: {result['config_name']}"


if __name__ == "__main__":
    main()
