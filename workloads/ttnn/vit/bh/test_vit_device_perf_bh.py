# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

sys.path.append(".")

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''

from loguru import logger  # noqa: E402

if not IS_POLARIS:
    import pytest  # type: ignore[import]  # noqa: F401, E402
    import torch  # type: ignore[no-redef]  # noqa: E402
    import ttnn  # type: ignore[no-redef, import]  # noqa: E402
    from models.demos.vision.classification.vit.common.tests.vit_test_infra import create_test_infra  # type: ignore[import]  # noqa: E402
    from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf  # type: ignore[import]  # noqa: E402
else:
    import ttsim.front.ttnn as ttnn  # type: ignore[no-redef]
    import ttsim.front.ttnn.minitorch_shim as torch  # type: ignore[no-redef]
    from ttsim.front.ttnn.device import set_default_device  # noqa: F401
    from workloads.ttnn.vit.bh.vit_test_infra_polaris_bh import create_test_infra  # type: ignore[no-redef]
    from workloads.ttnn.vit.bh.vit_polaris_params_bh import config_dict


# ---------------------------------------------------------------------------
# test_vit_device_ops — runs the full ViT graph (both modes)
# ---------------------------------------------------------------------------

def test_vit_device_ops(
    device,
    batch_size=10,
):
    torch.manual_seed(0)

    test_infra = create_test_infra(device, batch_size, use_random_input_tensor=True)

    if not IS_POLARIS:
        tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)
        tt_image_res = tt_inputs_host.to(device, sharded_mem_config_DRAM)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res)
        # include initial reshard in device perf test
        test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
        output_tensor = test_infra.run()
        # include final s2i in device perf test
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)
    else:
        tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)
        test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_host, input_mem_config)
        output_tensor = test_infra.run()
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)

        output = ttnn.to_torch(output_tensor)
        image_size = config_dict["image_size"]
        patch_size = config_dict["patch_size"]
        num_labels = 1152
        sequence_len = 1 + (image_size // patch_size) ** 2
        expected_output_shape = [batch_size, sequence_len, num_labels]
        assert output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, but got {output.shape}"
        )
        logger.info(f"test_vit_device_ops: obtained expected output shape {expected_output_shape}")
        return device


def run_vit_device_ops(wlname: str, device: ttnn.device.Device, cfg: dict):
    batch_size = cfg.get("bs", 10)
    return test_vit_device_ops(device, batch_size=batch_size)


def run_vit_perf_device(wlname: str, device: ttnn.device.Device, cfg: dict):
    """Workload-yaml entry point for analytical device perf reporting (BH)."""
    batch_size = cfg.get("bs", 10)
    test_vit_device_ops(device, batch_size=batch_size)
    test_vit_perf_device(batch_size=batch_size)


# ---------------------------------------------------------------------------
# test_vit_perf_device — dual-mode device profiling
# HW path: Tracy via subprocess to upstream pytest.
# Polaris path: analytical projection via run_device_perf_polaris.
# ---------------------------------------------------------------------------

def test_vit_perf_device(batch_size=10, expected_kernel_samples_per_sec=3050):
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    if IS_POLARIS:
        from workloads.common.polaris_device_perf import (
            run_device_perf_polaris,
            prep_device_perf_report_polaris,
        )

        post_processed_results = run_device_perf_polaris(
            test_fn=test_vit_device_ops,
            batch_size=batch_size,
            cols=cols,
            archspec="config/tt_bh.yaml",
            devname="p100a",
        )
        prep_device_perf_report_polaris(
            model_name=f"vit-bh-{batch_size}",
            batch_size=batch_size,
            post_processed_results=post_processed_results,
        )
        return

    command = (
        f"pytest models/demos/vision/classification/vit/blackhole/tests/"
        f"test_vit_device_perf_bh.py::test_vit_device_ops[{batch_size}-device_params0]"
    )

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    # BH-specific: num_iterations=2 to mitigate timeout (per BH HW comment).
    post_processed_results = run_device_perf(
        command, subdir="vit", num_iterations=2, cols=cols, batch_size=batch_size,
    )

    expected_results = check_device_perf(
        post_processed_results,
        margin=0.03,
        expected_perf_cols={inference_time_key: expected_kernel_samples_per_sec},
        # BH HW retains assert_on_fail=False from the original BH device-perf test.
        assert_on_fail=False,
    )
    prep_device_perf_report(
        model_name=f"vit-bh-{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )


# ---------------------------------------------------------------------------
# Registry and CLI
# ---------------------------------------------------------------------------

_STANDALONE_RUN_SPECS: list[tuple[str, object, str]] = [
    ("device-ops", run_vit_device_ops, "opt-vit-bh-device-ops"),
]

_STANDALONE_VALID_SHORT_NAMES = frozenset(s[0] for s in _STANDALONE_RUN_SPECS)


def run_one(callback, wlname: str, cfg: dict):
    if IS_POLARIS:
        from ttsim.front.ttnn.device import close_device, open_device
        device = open_device()
    else:
        from ttnn import close_device, open_device  # type: ignore[no-redef]
        device = open_device(device_id=0)
    callback(wlname, device, cfg)
    close_device(device)


def standalone(test_name: str | None = None) -> None:
    """Run standalone BH device-perf ViT tests, or a single test by short name."""
    all_names = _STANDALONE_VALID_SHORT_NAMES | {"device-perf"}

    if test_name == "device-perf":
        test_vit_perf_device()
        return

    if test_name is None:
        for _short, fn, wlname in _STANDALONE_RUN_SPECS:
            run_one(fn, wlname, {})
        return
    if test_name not in all_names:
        valid = ", ".join(sorted(all_names))
        logger.error(f"Unknown test {test_name}. Valid names: {valid}")
        sys.exit(1)
    for short, fn, wlname in _STANDALONE_RUN_SPECS:
        if short == test_name:
            run_one(fn, wlname, {})
            return


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    parser = argparse.ArgumentParser(
        description="Run optimized sharded ViT (Blackhole) device-perf standalone tests."
    )
    parser.add_argument(
        "test",
        nargs="?",
        metavar="TEST",
        default="device-ops",
        help=(
            "Run only this test by short name, "
            "e.g. device-ops, device-perf. If omitted, runs 'device-ops'."
        ),
    )
    _args = parser.parse_args()
    standalone(_args.test)
