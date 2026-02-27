#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Master test runner for all 5 Deformable DETR transformer modules.
Runs simple tests for each module and generates a summary report.
"""

import os
import sys
from datetime import datetime

# Add project root to path (go up 5 levels from deformable_transformer_shape/ to polaris/)
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    ),
)


class TeeWriter:
    """Write to multiple streams simultaneously (terminal + file)"""

    def __init__(self, *writers):
        self.writers = writers

    def write(self, text):
        for writer in self.writers:
            writer.write(text)
            writer.flush()

    def flush(self):
        for writer in self.writers:
            writer.flush()


# Import all test modules
from workloads.Deformable_DETR.tests.deformable_transformer.deformable_transformer_shape.test_encoder_layer_simple import (
    test_encoder_layer_with_numerical,
)
from workloads.Deformable_DETR.tests.deformable_transformer.deformable_transformer_shape.test_encoder_simple import (
    test_encoder,
)
from workloads.Deformable_DETR.tests.deformable_transformer.deformable_transformer_shape.test_decoder_layer_simple import (
    test_decoder_layer,
)
from workloads.Deformable_DETR.tests.deformable_transformer.deformable_transformer_shape.test_decoder_simple import (
    test_decoder,
)
from workloads.Deformable_DETR.tests.deformable_transformer.deformable_transformer_shape.test_full_transformer_simple import (
    test_full_transformer,
)


def run_all_tests():
    """Run all 5 module tests"""

    print("\n" + "=" * 80)
    print(" " * 20 + "DEFORMABLE DETR TRANSFORMER TEST SUITE")
    print("=" * 80)
    print("\nTesting 5 converted modules:")
    print("  1. DeformableTransformerEncoderLayer")
    print("  2. DeformableTransformerEncoder")
    print("  3. DeformableTransformerDecoderLayer")
    print("  4. DeformableTransformerDecoder")
    print("  5. DeformableTransformer (Full)")
    print("\nValidation Strategy:")
    print("  - Shape inference: Always tested (TTSim default behavior)")
    print("  - Numerical computation: Tested if data is available")
    print("=" * 80)

    results = []

    # Test 1: EncoderLayer
    print("\n\n")
    print("#" * 80)
    print("# TEST 1/5: DeformableTransformerEncoderLayer")
    print("#" * 80)
    try:
        success = test_encoder_layer_with_numerical()
        results.append(("EncoderLayer", success, None))
    except Exception as e:
        print(f"\n✗ FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("EncoderLayer", False, str(e)))

    # Test 2: Encoder
    print("\n\n")
    print("#" * 80)
    print("# TEST 2/5: DeformableTransformerEncoder")
    print("#" * 80)
    try:
        success = test_encoder()
        results.append(("Encoder", success, None))
    except Exception as e:
        print(f"\n✗ FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Encoder", False, str(e)))

    # Test 3: DecoderLayer
    print("\n\n")
    print("#" * 80)
    print("# TEST 3/5: DeformableTransformerDecoderLayer")
    print("#" * 80)
    try:
        success = test_decoder_layer()
        results.append(("DecoderLayer", success, None))
    except Exception as e:
        print(f"\n✗ FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("DecoderLayer", False, str(e)))

    # Test 4: Decoder
    print("\n\n")
    print("#" * 80)
    print("# TEST 4/5: DeformableTransformerDecoder")
    print("#" * 80)
    try:
        success = test_decoder()
        results.append(("Decoder", success, None))
    except Exception as e:
        print(f"\n✗ FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Decoder", False, str(e)))

    # Test 5: Full Transformer
    print("\n\n")
    print("#" * 80)
    print("# TEST 5/5: DeformableTransformer (Full)")
    print("#" * 80)
    try:
        success = test_full_transformer()
        results.append(("FullTransformer", success, None))
    except Exception as e:
        print(f"\n✗ FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("FullTransformer", False, str(e)))

    # Generate summary
    print("\n\n")
    print("=" * 80)
    print(" " * 30 + "TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    print(f"\nTotal Tests:  {len(results)}")
    print(f"Passed:       {passed} ✓")
    print(f"Failed:       {failed} ✗")
    print(f"Success Rate: {passed/len(results)*100:.1f}%")

    print("\n" + "-" * 80)
    print("Individual Results:")
    print("-" * 80)

    max_name_len = max(len(name) for name, _, _ in results)

    for name, success, error in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name:<{max_name_len}}  {status}")
        if error:
            error_short = error[:60] + "..." if len(error) > 60 else error
            print(f"    └─ Error: {error_short}")

    print("\n" + "=" * 80)

    if passed == len(results):
        print(" " * 25 + "ALL TESTS PASSED ✓✓✓")
    elif passed > 0:
        print(f" " * 20 + f"{passed}/{len(results)} TESTS PASSED")
    else:
        print(" " * 25 + "ALL TESTS FAILED ✗✗✗")

    print("=" * 80)

    return passed == len(results)


if __name__ == "__main__":
    # Setup output file
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    )
    report_dir = os.path.join(project_root, "workloads", "Deformable_DETR", "reports")
    os.makedirs(report_dir, exist_ok=True)

    report_file = os.path.join(report_dir, "deformable_transformer_shape_validation.md")

    # Redirect stdout to both terminal and file
    original_stdout = sys.stdout

    with open(report_file, "w", encoding="utf-8") as f:
        # Write markdown header
        f.write(f"# Deformable DETR Transformer Validation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write("```\n")

        # Redirect stdout to both terminal and file
        sys.stdout = TeeWriter(original_stdout, f)

        try:
            success = run_all_tests()
        finally:
            # Restore original stdout
            sys.stdout = original_stdout
            f.write("\n```\n")

    print(f"\n✓ Report saved to: {report_file}")
    sys.exit(0 if success else 1)
