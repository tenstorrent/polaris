#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Run all ScaledYOLOv4 YOLO module validations and generate results markdown.
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from workloads.ScaledYOLOv4.reference.yolo.model import validate_model_module
from workloads.ScaledYOLOv4.reference.yolo.detect import validate_detect_module

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


RTOL = 1e-4
ATOL = 1e-4


def generate_results_markdown(results: dict, output_path: str = None) -> str:
    """Generate markdown report from results dictionary."""
    total = len(results)

    # Count passes correctly for dict results
    passed_count = 0
    for v in results.values():
        if isinstance(v, dict):
            if v.get("passed", False):
                passed_count += 1
        elif v:
            passed_count += 1

    lines = [
        "# ScaledYOLOv4 YOLO Modules - ttsim vs PyTorch Validation Results",
        f"\nGenerated: {
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**{passed_count}/{total} modules passed** | Tolerances: rtol={RTOL}, atol={ATOL}",
        "\n---",
    ]

    for name, result in results.items():
        if isinstance(result, dict):
            passed = result.get("passed", False)
            status = "PASS" if passed else "FAIL"

            lines.append(f"\n## {name} - {status}")

            # Input info
            input_shape = result.get("input_shape", "N/A")
            input_values = result.get("input_values", "N/A")

            lines.append(f"\n**Input**")
            if (
                isinstance(input_shape, list)
                and len(input_shape) > 0
                and isinstance(input_shape[0], list)
            ):
                # Multi-scale inputs (Detect/Model)
                lines.append(f"- Shapes: `{input_shape}`")
            else:
                lines.append(f"- Shape: `{input_shape}`")
            lines.append(f"- Values: `{input_values}`")

            # Config info (if available)
            if "config" in result:
                lines.append(f"\n**Configuration**")
                lines.append(f"- Config file: `{result.get('config', 'N/A')}`")
            if "num_layers" in result:
                lines.append(f"- Number of layers: {result.get('num_layers', 'N/A')}")
            if "weights_transferred" in result:
                lines.append(
                    f"- Weights transferred: {result.get('weights_transferred', 'N/A')}"
                )

            # Output info
            pytorch_shape = result.get("pytorch_output_shape", "N/A")
            ttsim_shape = result.get("ttsim_output_shape", "N/A")
            pytorch_values = result.get("pytorch_output_values", "N/A")
            ttsim_values = result.get("ttsim_output_values", "N/A")

            lines.append(f"\n**Output Shapes**")
            if (
                isinstance(pytorch_shape, list)
                and len(pytorch_shape) > 0
                and isinstance(pytorch_shape[0], list)
            ):
                # Multi-scale outputs
                for i, (pt_s, tt_s) in enumerate(zip(pytorch_shape, ttsim_shape)):
                    lines.append(f"- Layer {i}: PyTorch `{pt_s}`, ttsim `{tt_s}`")
            else:
                lines.append(f"- PyTorch: `{pytorch_shape}`")
                lines.append(f"- ttsim: `{ttsim_shape}`")

            lines.append(f"\n**Output Values**")
            if isinstance(pytorch_values, list):
                for i, (pt_v, tt_v) in enumerate(zip(pytorch_values, ttsim_values)):
                    lines.append(f"- Layer {i}:")
                    lines.append(f"  - PyTorch: `{pt_v}`")
                    lines.append(f"  - ttsim: `{tt_v}`")
            else:
                lines.append(f"- PyTorch: `{pytorch_values}`")
                lines.append(f"- ttsim: `{ttsim_values}`")

            # Per-layer results (if available)
            if "layers" in result and result["layers"]:
                layers = result["layers"]
                passed_layers = sum(1 for l in layers if l.get("passed", False))
                lines.append(
                    f"\n**Per-Layer Results ({passed_layers}/{len(layers)} passed)**"
                )
                lines.append("")
                lines.append(
                    "| Layer | Type | PyTorch Shape | ttsim Shape | Max Diff | Status |"
                )
                lines.append(
                    "|-------|------|---------------|-------------|----------|--------|"
                )

                for layer_result in layers:
                    layer_idx = layer_result.get("layer", "?")
                    layer_type = layer_result.get("type", "Unknown")
                    pt_shape = layer_result.get("pytorch_shape", "N/A")
                    tt_shape = layer_result.get("ttsim_shape", "N/A")
                    layer_passed = layer_result.get("passed", False)
                    max_diff = layer_result.get("max_diff", None)
                    shape_only = layer_result.get("shape_only", False)

                    if shape_only:
                        status = "✓ (shape-only)"
                        diff_str = "N/A"
                    elif layer_passed:
                        status = "✓"
                        diff_str = f"{
                            max_diff:.2e}" if max_diff is not None else "N/A"
                    else:
                        status = "✗"
                        reason = layer_result.get("reason", "")
                        diff_str = f"{
                            max_diff:.2e}" if max_diff is not None else reason

                    # Truncate type name for readability
                    layer_type_short = (
                        layer_type[:20] if len(layer_type) > 20 else layer_type
                    )
                    lines.append(
                        f"| {layer_idx} | {layer_type_short} | `{pt_shape}` | `{tt_shape}` | {diff_str} | {status} |"
                    )

            # Comparison summary
            if "values_match" in result:
                values_match = result.get("values_match", None)
                max_diff = result.get("max_diff", None)
                mean_diff = result.get("mean_diff", None)

                lines.append(f"\n**Comparison**")
                if values_match is not None:
                    match_str = "Yes" if values_match else "No"
                    lines.append(f"- Values Match: {match_str}")
                if max_diff is not None:
                    lines.append(f"- Max Diff: {max_diff:.2e}")
                if mean_diff is not None:
                    lines.append(f"- Mean Diff: {mean_diff:.2e}")

            # Error info (if any)
            if "error" in result:
                lines.append(f"\n**Error**")
                lines.append(f"- {result.get('error')}")
        else:
            status = "PASS" if result else "FAIL"
            lines.append(f"\n## {name} - {status}")

        lines.append("\n---")

    md_content = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"\nResults written to: {output_path}")

    return md_content


def run_all_validations(verbose: bool = False):
    """Run all YOLO module validations."""
    results = {}

    print("\n" + "=" * 60)
    print("ScaledYOLOv4 YOLO Modules - ttsim vs PyTorch Validation")
    print("=" * 60)

    # Run each validation
    validations = [
        ("Detect", lambda v: validate_detect_module(verbose=v)),
        ("Model", lambda v: validate_model_module(verbose=v)),
    ]

    for name, validate_fn in validations:
        print(f"\n{'=' * 60}")
        print(f"Testing: {name}")
        print("=" * 60)
        try:
            results[name] = validate_fn(verbose)
        except Exception as e:
            import traceback

            print(f"\nERROR in {name}: {e}")
            traceback.print_exc()
            results[name] = {"passed": False, "error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = 0
    for name, result in results.items():
        if isinstance(result, dict):
            passed_flag = result.get("passed", False)
        else:
            passed_flag = result
        if passed_flag:
            passed += 1
        status = "PASS" if passed_flag else "FAIL"
        print(f"  {name}: {status}")
    total = len(results)
    print(f"\n{passed}/{total} modules passed")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ScaledYOLOv4 YOLO module validations"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output markdown file path (default: yolo_results_YYYYMMDD_HHMMSS.md)",
    )
    args = parser.parse_args()

    results = run_all_validations(verbose=args.verbose)

    # Generate markdown report
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(__file__), f"yolo_results_{timestamp}.md"
        )

    generate_results_markdown(results, output_path)

    # Exit with appropriate code
    all_passed = all(
        r.get("passed", False) if isinstance(r, dict) else r for r in results.values()
    )
    sys.exit(0 if all_passed else 1)
