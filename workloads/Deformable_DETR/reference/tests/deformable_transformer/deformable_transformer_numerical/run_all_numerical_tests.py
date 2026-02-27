#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Master runner for all Deformable Transformer numerical validation tests.

Runs all 5 test modules and prints a summary table.

Tests:
  1. EncoderLayer  — single encoder layer numerical comparison
  2. Encoder       — stacked encoder layers
  3. DecoderLayer  — single decoder layer (with norm swap)
  4. Decoder       — stacked decoder layers
  5. FullTransform — complete encoder-decoder pipeline
"""

import os
import sys
import time
import traceback

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    ),
)

from workloads.Deformable_DETR.tests.deformable_transformer.deformable_transformer_numerical.test_encoder_layer_numerical import (
    test_encoder_layer_numerical,
)
from workloads.Deformable_DETR.tests.deformable_transformer.deformable_transformer_numerical.test_encoder_numerical import (
    test_encoder_numerical,
)
from workloads.Deformable_DETR.tests.deformable_transformer.deformable_transformer_numerical.test_decoder_layer_numerical import (
    test_decoder_layer_numerical,
)
from workloads.Deformable_DETR.tests.deformable_transformer.deformable_transformer_numerical.test_decoder_numerical import (
    test_decoder_numerical,
)
from workloads.Deformable_DETR.tests.deformable_transformer.deformable_transformer_numerical.test_full_transformer_numerical import (
    test_full_transformer_numerical,
)


def run_all_numerical_tests():
    """Run all numerical validation tests and print summary."""

    tests = [
        ("1. EncoderLayer", test_encoder_layer_numerical),
        ("2. Encoder (stacked)", test_encoder_numerical),
        ("3. DecoderLayer", test_decoder_layer_numerical),
        ("4. Decoder (stacked)", test_decoder_numerical),
        ("5. Full Transformer", test_full_transformer_numerical),
    ]

    results = []
    total_time = 0

    for name, test_fn in tests:
        print(f"\n{'#' * 80}")
        print(f"# Running: {name}")
        print(f"{'#' * 80}")
        t0 = time.time()
        try:
            passed = test_fn()
            elapsed = time.time() - t0
            results.append((name, "PASS" if passed else "FAIL", elapsed, None))
        except Exception as e:
            elapsed = time.time() - t0
            results.append((name, "ERROR", elapsed, str(e)))
            traceback.print_exc()
        total_time += elapsed

    # ── Summary ──
    print("\n")
    print("=" * 80)
    print("  NUMERICAL VALIDATION SUMMARY — Deformable Transformer")
    print("=" * 80)
    print(f"\n  {'Test':<30s} {'Result':<10s} {'Time':>8s}")
    print(f"  {'-'*30} {'-'*10} {'-'*8}")

    pass_count = 0
    for name, status, elapsed, err in results:
        mark = "✓" if status == "PASS" else "✗"
        print(f"  {name:<30s} {mark} {status:<8s} {elapsed:>7.2f}s")
        if status == "PASS":
            pass_count += 1
        if err:
            print(f"    Error: {err}")

    total = len(tests)
    print(f"\n  {'-'*50}")
    print(f"  Total: {pass_count}/{total} passed   ({total_time:.2f}s)")
    print("=" * 80)

    return pass_count == total


class _TeeStream:
    """Write to both a file and the original stream simultaneously."""

    def __init__(self, original, filepath):
        self._original = original
        self._file = open(filepath, "w", encoding="utf-8")

    def write(self, text):
        self._original.write(text)
        self._file.write(text)

    def flush(self):
        self._original.flush()
        self._file.flush()

    def close(self):
        self._file.close()


if __name__ == "__main__":
    REPORT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports")
    )
    os.makedirs(REPORT_DIR, exist_ok=True)
    REPORT_PATH = os.path.join(
        REPORT_DIR, "deformable_transformer_numerical_validation.md"
    )

    tee = _TeeStream(sys.stdout, REPORT_PATH)
    sys.stdout = tee

    success = run_all_numerical_tests()
    print(f"\n\n*Report saved to: {REPORT_PATH}*")
    tee.close()
    sys.stdout = tee._original
    sys.exit(0 if success else 1)
