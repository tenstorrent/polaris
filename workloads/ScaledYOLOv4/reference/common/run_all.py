#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Run all ScaledYOLOv4 module validations and generate results markdown.
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from workloads.ScaledYOLOv4.reference.common.conv import validate_conv_module
from workloads.ScaledYOLOv4.reference.common.bottleneck import (
    validate_bottleneck_module,
)
from workloads.ScaledYOLOv4.reference.common.bottleneck_csp import (
    validate_bottleneck_csp_module,
)
from workloads.ScaledYOLOv4.reference.common.bottleneck_csp2 import (
    validate_bottleneck_csp2_module,
)
from workloads.ScaledYOLOv4.reference.common.dwconv_layer import (
    validate_dwconv_layer_module,
)
from workloads.ScaledYOLOv4.reference.common.dwconv import validate_dwconv_module
from workloads.ScaledYOLOv4.reference.common.conv_layer import (
    validate_conv_layer_module,
)
from workloads.ScaledYOLOv4.reference.common.comb_conv_layer import (
    validate_comb_conv_layer_module,
)
from workloads.ScaledYOLOv4.reference.common.upsample import validate_upsample_module
from workloads.ScaledYOLOv4.reference.common.maxpool import validate_maxpool_module
from workloads.ScaledYOLOv4.reference.common.concat import validate_concat_module
from workloads.ScaledYOLOv4.reference.common.flatten import validate_flatten_module
from workloads.ScaledYOLOv4.reference.common.classify import validate_classify_module
from workloads.ScaledYOLOv4.reference.common.focus import validate_focus_module
from workloads.ScaledYOLOv4.reference.common.spp import validate_spp_module
from workloads.ScaledYOLOv4.reference.common.sppcsp import validate_sppcsp_module
from workloads.ScaledYOLOv4.reference.common.vovcsp import validate_vovcsp_module
from workloads.ScaledYOLOv4.reference.common.convsig import validate_convsig_module

RTOL = 1e-5
ATOL = 1e-6


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
        "# ScaledYOLOv4 ttsim vs PyTorch Validation Results",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**{passed_count}/{total} modules passed** | Tolerances: rtol={RTOL}, atol={ATOL}",
        "\n---",
    ]

    for name, result in results.items():
        if isinstance(result, dict):
            passed = result.get("passed", False)
            status = "PASS" if passed else "FAIL"
            input_shape = result.get("input_shape", "N/A")
            pytorch_shape = result.get("pytorch_output_shape", "N/A")
            ttsim_shape = result.get("ttsim_output_shape", "N/A")
            shape_only = result.get("shape_only", False)
            values_match = result.get("values_match", None)
            max_diff = result.get("max_diff", None)
            mean_diff = result.get("mean_diff", None)
            input_values = result.get("input_values", "N/A")
            pytorch_values = result.get("pytorch_output_values", "N/A")
            ttsim_values = result.get("ttsim_output_values", "N/A")

            lines.append(f"\n## {name} - {status}")

            # Input
            lines.append(f"\n**Input**")
            lines.append(f"- Shape: `{input_shape}`")
            lines.append(f"- Values: `{input_values}`")

            # Shape comparison
            lines.append(f"\n**Shape**")
            lines.append(f"- PyTorch: `{pytorch_shape}`")
            lines.append(f"- ttsim: `{ttsim_shape}`")
            shape_match = pytorch_shape == ttsim_shape
            lines.append(f"- Match: {'Yes' if shape_match else 'No'}")

            # Values comparison
            lines.append(f"\n**Output Values**")
            lines.append(f"- PyTorch: `{pytorch_values}`")
            lines.append(f"- ttsim: `{ttsim_values}`")
            if shape_only:
                lines.append(f"- Match: N/A (shape-only)")
            elif values_match is not None:
                match_str = "Yes" if values_match else "No"
                lines.append(
                    f"- Match: {match_str} (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})"
                )
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
    """Run all module validations."""
    results = {}

    print("\n" + "=" * 50)
    print("ScaledYOLOv4 ttsim vs PyTorch Validation")
    print("=" * 50)

    # Run each validation
    validations = [
        ("Conv", validate_conv_module),
        ("DWConv", validate_dwconv_module),
        ("Bottleneck", validate_bottleneck_module),
        ("BottleneckCSP", validate_bottleneck_csp_module),
        ("BottleneckCSP2", validate_bottleneck_csp2_module),
        ("DWConvLayer", validate_dwconv_layer_module),
        ("ConvLayer", validate_conv_layer_module),
        ("CombConvLayer", validate_comb_conv_layer_module),
        ("Upsample", validate_upsample_module),
        ("MaxPool", validate_maxpool_module),
        ("Concat", validate_concat_module),
        ("Flatten", validate_flatten_module),
        ("Classify", validate_classify_module),
        ("Focus", validate_focus_module),
        ("SPP", validate_spp_module),
        ("SPPCSP", validate_sppcsp_module),
        ("VoVCSP", validate_vovcsp_module),
        ("ConvSig", validate_convsig_module),
    ]

    for name, validate_fn in validations:
        try:
            results[name] = validate_fn(verbose=verbose)
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
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
    parser = argparse.ArgumentParser(description="Run ScaledYOLOv4 module validations")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show intermediate tensor values"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output markdown file path (default: results_YYYYMMDD_HHMMSS.md)",
    )
    args = parser.parse_args()

    results = run_all_validations(verbose=args.verbose)

    # Generate markdown report
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(os.path.dirname(__file__), f"results_{timestamp}.md")

    generate_results_markdown(results, output_path)

    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)
