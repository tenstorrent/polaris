#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Test file for ms_deform_attn_core: Shape inference and numerical validation.
Compares TTSim implementation against PyTorch reference module-by-module.
"""

import sys
import os
import numpy as np
import pytest
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
from workloads.Deformable_DETR.models.ops.functions.ms_deform_attn_func_ttsim import (
    ms_deform_attn_core_ttsim,
)


def ms_deform_attn_core_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """PyTorch reference implementation."""
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = (
            value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        )
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        N_ * M_, 1, Lq_, L_ * P_
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(N_, M_ * D_, Lq_)
    )
    return output.transpose(1, 2).contiguous()


def pytorch_with_intermediates(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """PyTorch with intermediate outputs for comparison."""
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape

    intermediates = {}

    # Step 1: Split
    split_sizes = [H_ * W_ for H_, W_ in value_spatial_shapes]
    value_list = value.split(split_sizes, dim=1)
    intermediates["value_list_shapes"] = [list(v.shape) for v in value_list]
    intermediates["value_list"] = [v.detach().cpu().numpy() for v in value_list]

    # Step 2: Sampling grids
    sampling_grids = 2 * sampling_locations - 1
    intermediates["sampling_grids_shape"] = list(sampling_grids.shape)
    intermediates["sampling_grids"] = sampling_grids.detach().cpu().numpy()

    # Step 3-4: Per-level processing
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = (
            value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        )
        intermediates[f"level_{lid_}_value_reshaped_shape"] = list(value_l_.shape)
        intermediates[f"level_{lid_}_value_reshaped"] = value_l_.detach().cpu().numpy()

        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        intermediates[f"level_{lid_}_sampling_grid_shape"] = list(
            sampling_grid_l_.shape
        )
        intermediates[f"level_{lid_}_sampling_grid"] = (
            sampling_grid_l_.detach().cpu().numpy()
        )

        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        intermediates[f"level_{lid_}_grid_sample_output_shape"] = list(
            sampling_value_l_.shape
        )
        intermediates[f"level_{lid_}_grid_sample_output"] = (
            sampling_value_l_.detach().cpu().numpy()
        )
        sampling_value_list.append(sampling_value_l_)

    # Step 5: Stack
    stacked = torch.stack(sampling_value_list, dim=-2)
    intermediates["stacked_shape"] = list(stacked.shape)
    intermediates["stacked"] = stacked.detach().cpu().numpy()

    # Step 6: Flatten
    flattened = stacked.flatten(-2)
    intermediates["flattened_shape"] = list(flattened.shape)
    intermediates["flattened"] = flattened.detach().cpu().numpy()

    # Step 7: Attention reshape
    attn_reshaped = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    intermediates["attention_reshaped_shape"] = list(attn_reshaped.shape)
    intermediates["attention_reshaped"] = attn_reshaped.detach().cpu().numpy()

    # Step 8: Weighted sum
    weighted = (flattened * attn_reshaped).sum(-1)
    intermediates["weighted_shape"] = list(weighted.shape)
    intermediates["weighted"] = weighted.detach().cpu().numpy()

    # Step 9-11: Final reshape and transpose
    output = weighted.view(N_, M_ * D_, Lq_).transpose(1, 2).contiguous()
    intermediates["output_shape"] = list(output.shape)
    intermediates["output"] = output.detach().cpu().numpy()

    return output, intermediates


def ttsim_with_intermediates(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """TTSim with intermediate outputs for comparison."""
    from types import SimpleNamespace
    from ttsim.ops.desc.nn import grid_sample_fwd

    N_, S_, M_, D_ = value.shape
    _, Lq_, _, L_, P_, _ = sampling_locations.shape

    intermediates = {}

    # Step 1: Split
    split_sizes = [int(H * W) for H, W in value_spatial_shapes]
    value_list = np.split(value, np.cumsum(split_sizes)[:-1], axis=1)
    intermediates["value_list_shapes"] = [list(v.shape) for v in value_list]
    intermediates["value_list"] = value_list

    # Step 2: Sampling grids
    sampling_grids = 2.0 * sampling_locations - 1.0
    intermediates["sampling_grids_shape"] = list(sampling_grids.shape)
    intermediates["sampling_grids"] = sampling_grids

    # Step 3-4: Per-level processing
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        H_, W_ = int(H_), int(W_)

        value_l_data = value_list[lid_]
        value_l_flat = value_l_data.reshape(N_, H_ * W_, M_ * D_)
        value_l_trans = value_l_flat.transpose(0, 2, 1)
        value_l_reshaped = value_l_trans.reshape(N_ * M_, D_, H_, W_)
        intermediates[f"level_{lid_}_value_reshaped_shape"] = list(
            value_l_reshaped.shape
        )
        intermediates[f"level_{lid_}_value_reshaped"] = value_l_reshaped

        sampling_grid_l_data = sampling_grids[:, :, :, lid_]
        sampling_grid_l_trans = sampling_grid_l_data.transpose(0, 2, 1, 3, 4)
        sampling_grid_l_flat = sampling_grid_l_trans.reshape(N_ * M_, Lq_, P_, 2)
        intermediates[f"level_{lid_}_sampling_grid_shape"] = list(
            sampling_grid_l_flat.shape
        )
        intermediates[f"level_{lid_}_sampling_grid"] = sampling_grid_l_flat

        input_t = SimpleNamespace(
            shape=list(value_l_reshaped.shape),
            data=value_l_reshaped,
            dtype=value_l_reshaped.dtype,
        )
        grid_t = SimpleNamespace(
            shape=list(sampling_grid_l_flat.shape),
            data=sampling_grid_l_flat,
            dtype=sampling_grid_l_flat.dtype,
        )
        output_t = SimpleNamespace(shape=None, data=None, dtype=None)
        op = SimpleNamespace(
            attrs={"mode": "bilinear", "padding_mode": "zeros", "align_corners": False},
            optype="GridSample",
        )

        grid_sample_fwd([input_t, grid_t], [output_t], op)
        sampling_value_l_ = output_t.data
        intermediates[f"level_{lid_}_grid_sample_output_shape"] = list(
            sampling_value_l_.shape
        )
        intermediates[f"level_{lid_}_grid_sample_output"] = sampling_value_l_
        sampling_value_list.append(sampling_value_l_)

    # Step 5: Stack
    stacked = np.stack(sampling_value_list, axis=-2)
    intermediates["stacked_shape"] = list(stacked.shape)
    intermediates["stacked"] = stacked

    # Step 6: Flatten
    flattened = stacked.reshape(N_ * M_, D_, Lq_, L_ * P_)
    intermediates["flattened_shape"] = list(flattened.shape)
    intermediates["flattened"] = flattened

    # Step 7: Attention reshape
    attn_trans = attention_weights.transpose(0, 2, 1, 3, 4)
    attn_reshaped = attn_trans.reshape(N_ * M_, 1, Lq_, L_ * P_)
    intermediates["attention_reshaped_shape"] = list(attn_reshaped.shape)
    intermediates["attention_reshaped"] = attn_reshaped

    # Step 8: Weighted sum
    weighted = (flattened * attn_reshaped).sum(axis=-1)
    intermediates["weighted_shape"] = list(weighted.shape)
    intermediates["weighted"] = weighted

    # Step 9-11: Final reshape and transpose
    output_viewed = weighted.reshape(N_, M_ * D_, Lq_)
    output_transposed = output_viewed.transpose(0, 2, 1)
    output_final = np.ascontiguousarray(output_transposed)
    intermediates["output_shape"] = list(output_final.shape)
    intermediates["output"] = output_final

    return output_final, intermediates


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
    """Test shape inference mode."""
    print("\n" + "=" * 80)
    print("SHAPE INFERENCE TEST")
    print("=" * 80)

    N, Lq, M, D = 2, 100, 8, 32
    L, P = 4, 4
    spatial_shapes = np.array([[50, 50], [25, 25], [13, 13], [7, 7]], dtype=np.int32)
    S = int(np.sum(spatial_shapes[:, 0] * spatial_shapes[:, 1]))

    value_sim = SimTensor(
        {"name": "value", "shape": [N, S, M, D], "data": None, "dtype": np.dtype("float32")}
    )
    sampling_locations_sim = SimTensor(
        {
            "name": "sampling_locations",
            "shape": [N, Lq, M, L, P, 2],
            "data": None,
            "dtype": np.dtype("float32"),
        }
    )
    attention_weights_sim = SimTensor(
        {
            "name": "attention_weights",
            "shape": [N, Lq, M, L, P],
            "data": None,
            "dtype": np.dtype("float32"),
        }
    )

    output_sim = ms_deform_attn_core_ttsim(
        value_sim, spatial_shapes, sampling_locations_sim, attention_weights_sim
    )

    expected_shape = [N, Lq, M * D]
    actual_shape = output_sim.shape

    result = {
        "test": "Shape Inference",
        "config": f"N={N}, Lq={Lq}, M={M}, D={D}, L={L}, P={P}",
        "expected_shape": expected_shape,
        "actual_shape": actual_shape,
        "passed": expected_shape == actual_shape,
    }

    print(f"Expected shape: {expected_shape}")
    print(f"Actual shape: {actual_shape}")
    print(f"Test: {'PASSED' if result['passed'] else 'FAILED'}")

    return result


@pytest.mark.parametrize(
    "config_name,N,Lq,M,D,L,P,spatial_shapes",
    [
        (
            "Small Configuration",
            2, 100, 8, 32, 4, 4,
            np.array([[50, 50], [25, 25], [13, 13], [7, 7]], dtype=np.int32),
        ),
        (
            "Single Level",
            1, 50, 4, 16, 1, 2,
            np.array([[32, 32]], dtype=np.int32),
        ),
        (
            "Multiple Sampling Points",
            1, 64, 4, 16, 2, 8,
            np.array([[16, 16], [8, 8]], dtype=np.int32),
        ),
    ],
    ids=["small_config", "single_level", "multi_points"],
)
def test_numerical_computation(config_name, N, Lq, M, D, L, P, spatial_shapes):
    """Test numerical computation with detailed intermediate comparisons."""
    print(f"\n{'='*80}")
    print(f"NUMERICAL TEST - {config_name}")
    print(f"{'='*80}")
    print(f"Config: N={N}, Lq={Lq}, M={M}, D={D}, L={L}, P={P}")
    print(f"Spatial shapes: {spatial_shapes.tolist()}")

    S = int(np.sum(spatial_shapes[:, 0] * spatial_shapes[:, 1]))

    # Generate test data
    np.random.seed(42)
    value_np = np.random.randn(N, S, M, D).astype(np.float32)
    sampling_locations_np = np.random.rand(N, Lq, M, L, P, 2).astype(np.float32)
    attention_weights_np = np.random.rand(N, Lq, M, L, P).astype(np.float32)
    attention_weights_np = attention_weights_np / attention_weights_np.sum(
        axis=-1, keepdims=True
    )

    # PyTorch computation
    value_torch = torch.from_numpy(value_np)
    sampling_locations_torch = torch.from_numpy(sampling_locations_np)
    attention_weights_torch = torch.from_numpy(attention_weights_np)
    spatial_shapes_torch = torch.from_numpy(spatial_shapes)

    output_torch, pytorch_inter = pytorch_with_intermediates(
        value_torch,
        spatial_shapes_torch,
        sampling_locations_torch,
        attention_weights_torch,
    )

    # TTSim computation
    output_ttsim, ttsim_inter = ttsim_with_intermediates(
        value_np, spatial_shapes, sampling_locations_np, attention_weights_np
    )

    # Compare intermediates
    results = []
    tolerance = {"max_abs": 1e-4, "mean_abs": 1e-5, "relative": 1e-3}

    # Compare each intermediate step
    comparisons = [
        ("sampling_grids", "Sampling Grids [0,1] -> [-1,1]"),
        ("stacked", "Stacked Grid Sample Outputs"),
        ("flattened", "Flattened Stacked Outputs"),
        ("attention_reshaped", "Reshaped Attention Weights"),
        ("weighted", "Weighted Sum"),
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

            # Sample values for display
            pytorch_sample = format_array_sample(
                pytorch_val[:1, :1, :5]
                if pytorch_val.ndim >= 3
                else pytorch_val.flatten()[:10]
            )
            ttsim_sample = format_array_sample(
                ttsim_val[:1, :1, :5]
                if ttsim_val.ndim >= 3
                else ttsim_val.flatten()[:10]
            )

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

    # Per-level grid sample comparison
    print(f"\nPer-Level Grid Sample Comparison:")
    for lid in range(L):
        key = f"level_{lid}_grid_sample_output"
        if key in pytorch_inter and key in ttsim_inter:
            pytorch_val = pytorch_inter[key]
            ttsim_val = ttsim_inter[key]

            metrics = compute_error_metrics(pytorch_val, ttsim_val)

            passed = (
                metrics["max_abs_error"] < tolerance["max_abs"]
                and metrics["mean_abs_error"] < tolerance["mean_abs"]
                and metrics["relative_error"] < tolerance["relative"]
            )

            pytorch_sample = format_array_sample(pytorch_val.flatten()[:10])
            ttsim_sample = format_array_sample(ttsim_val.flatten()[:10])

            results.append(
                {
                    "module": f"Level {lid} Grid Sample",
                    "shape": list(pytorch_val.shape),
                    "metrics": metrics,
                    "pytorch_sample": pytorch_sample,
                    "ttsim_sample": ttsim_sample,
                    "passed": passed,
                }
            )

            print(f"  Level {lid}:")
            print(f"    Shape: {list(pytorch_val.shape)}")
            print(f"    PyTorch sample: {pytorch_sample}")
            print(f"    TTSim sample: {ttsim_sample}")
            print(
                f"    Max error={metrics['max_abs_error']:.6e}, "
                f"Rel error={metrics['relative_error']:.6e}, "
                f"{'PASSED' if passed else 'FAILED'}"
            )

    return {
        "config_name": config_name,
        "config": {
            "N": N,
            "Lq": Lq,
            "M": M,
            "D": D,
            "L": L,
            "P": P,
            "spatial_shapes": spatial_shapes.tolist(),
        },
        "tolerance": tolerance,
        "results": results,
        "overall_passed": all(r["passed"] for r in results),
    }


def generate_markdown_report(shape_result, numerical_results, output_path):
    """Generate markdown report."""

    md = []
    md.append(f"# MS Deform Attention Core - Test Report")
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
        for r in test["results"][:6]:  # Show samples for main comparisons
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

    # Numerical computation tests with different configurations
    numerical_results = []

    # Test 1: Small config
    numerical_results.append(
        test_numerical_computation(
            "Small Configuration",
            N=2,
            Lq=100,
            M=8,
            D=32,
            L=4,
            P=4,
            spatial_shapes=np.array(
                [[50, 50], [25, 25], [13, 13], [7, 7]], dtype=np.int32
            ),
        )
    )

    # Test 2: Single level
    numerical_results.append(
        test_numerical_computation(
            "Single Level",
            N=1,
            Lq=50,
            M=4,
            D=16,
            L=1,
            P=2,
            spatial_shapes=np.array([[32, 32]], dtype=np.int32),
        )
    )

    # Test 3: Multiple points
    numerical_results.append(
        test_numerical_computation(
            "Multiple Sampling Points",
            N=1,
            Lq=64,
            M=4,
            D=16,
            L=2,
            P=8,
            spatial_shapes=np.array([[16, 16], [8, 8]], dtype=np.int32),
        )
    )

    # Generate report
    reports_dir = os.path.join(_root, "workloads", "Deformable_DETR", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    output_path = os.path.join(reports_dir, f"ms_deform_attn_func_validation.md")

    generate_markdown_report(shape_result, numerical_results, output_path)


if __name__ == "__main__":
    main()
