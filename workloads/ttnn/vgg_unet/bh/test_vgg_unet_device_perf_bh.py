# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

sys.path.append('.')

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''

from loguru import logger  # noqa: E402

if not IS_POLARIS:
    import time  # noqa: E402
    import torch  # type: ignore[import-not-found]  # noqa: E402
    import ttnn  # type: ignore[no-redef, import]  # noqa: E402
    from models.demos.vision.segmentation.vgg_unet.common.runner.performant_runner import (  # type: ignore[import]  # noqa: E402
        VggUnetTrace2CQ,
    )
    from models.perf.device_perf_utils import (  # type: ignore[import]  # noqa: E402
        check_device_perf,
        prep_device_perf_report,
        run_device_perf,
    )
else:
    import ttsim.front.ttnn as ttnn  # type: ignore[no-redef]
    import ttsim.front.ttnn.minitorch_shim as torch  # type: ignore[no-redef]
    from workloads.ttnn.vgg_unet.bh.vgg_unet_test_infra_polaris_bh import VggUnetTrace2CQ  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# run_vgg_unet_e2e — mirrors run_vgg_unet_e2e from test_e2e_performant.py
# ---------------------------------------------------------------------------

def run_vgg_unet_e2e(device, batch_size: int = 1):
    # `batch_size` matches the (device, batch_size)->device contract documented on
    # run_device_perf_polaris; pass-through to the inner trace as device_batch_size.
    vgg_unet_trace_2cq = VggUnetTrace2CQ()
    vgg_unet_trace_2cq.initialize_vgg_unet_trace_2cqs_inference(
        device,
        model_location_generator=None,
        device_batch_size=batch_size,
    )

    if not IS_POLARIS:
        batch_size = batch_size * device.get_num_devices()
        input_shape = (batch_size, 3, 256, 256)
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        inference_iter_count = 3
        t0 = time.time()
        for _ in range(inference_iter_count):
            output = vgg_unet_trace_2cq.run(torch_input_tensor)
        ttnn.synchronize_device(device)
        t1 = time.time()
        vgg_unet_trace_2cq.release_vgg_unet_trace_2cqs_inference()
        inference_time_avg = round((t1 - t0) / inference_iter_count, 6)
        logger.info(
            f'ttnn_vgg_unet_256x256_batch_size_{batch_size}. '
            f'One inference iteration time (sec): {inference_time_avg}, '
            f'FPS: {round(batch_size / inference_time_avg)}'
        )
    else:
        output = vgg_unet_trace_2cq.run()
        vgg_unet_trace_2cq.release_vgg_unet_trace_2cqs_inference()
        output_torch = ttnn.to_torch(output)
        expected_shape = [batch_size, 1, 256, 256]
        assert output_torch.shape == expected_shape, (
            f'Expected output shape {expected_shape}, but got {output_torch.shape}'
        )
        logger.info(f'run_vgg_unet_e2e: obtained expected output shape {expected_shape}')

    return device


def run_vgg_unet_e2e_entry(wlname: str, device: ttnn.device.Device, cfg: dict):
    batch_size = cfg.get('bs', 1)
    return run_vgg_unet_e2e(device, batch_size=batch_size)


def run_vgg_unet_perf_device(wlname: str, device: ttnn.device.Device, cfg: dict):
    """Workload-yaml entry point for analytical device perf reporting (BH).

    Builds the graph on the polproj-provided *device* (so polproj's back-end
    pipeline can run its own stats), and also produces a standalone JSON perf
    report via the Polaris analytical path.
    """
    batch_size = cfg.get('bs', 1)
    run_vgg_unet_e2e(device, batch_size=batch_size)
    test_vgg_unet_perf_device(batch_size=batch_size)


# ---------------------------------------------------------------------------
# test_vgg_unet_perf_device — dual-mode device profiling
# HW path: Tracy via subprocess wrapping test_e2e_performant.py.
# Polaris path: analytical projection via run_device_perf_polaris (BH arch).
# ---------------------------------------------------------------------------

def test_vgg_unet_perf_device(batch_size: int = 1, expected_kernel_samples_per_sec: int = 320):
    cols = ['DEVICE FW', 'DEVICE KERNEL', 'DEVICE BRISC KERNEL']

    if IS_POLARIS:
        from workloads.common.polaris_device_perf import (
            run_device_perf_polaris,
            prep_device_perf_report_polaris,
        )

        post_processed_results = run_device_perf_polaris(
            test_fn=run_vgg_unet_e2e,
            batch_size=batch_size,
            cols=cols,
            archspec='config/tt_bh.yaml',
            devname='p100a',
        )
        prep_device_perf_report_polaris(
            model_name=f'vgg-unet-bh-{batch_size}',
            batch_size=batch_size,
            post_processed_results=post_processed_results,
        )
        return

    command = (
        f'pytest models/demos/vision/segmentation/vgg_unet/blackhole/tests/perf/'
        f'test_e2e_performant.py::test_vgg_unet_e2e[{batch_size}-device_params0]'
    )

    inference_time_key = 'AVG DEVICE KERNEL SAMPLES/S'
    # BH: num_iterations=2 to mitigate timeout (consistent with BH ViT pattern).
    post_processed_results = run_device_perf(
        command, subdir='vgg_unet', num_iterations=2, cols=cols, batch_size=batch_size,
    )

    expected_results = check_device_perf(
        post_processed_results,
        margin=0.03,
        expected_perf_cols={inference_time_key: expected_kernel_samples_per_sec},
        # BH HW retains assert_on_fail=False from the original BH device-perf test.
        assert_on_fail=False,
    )
    prep_device_perf_report(
        model_name=f'vgg-unet-bh-{batch_size}',
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments='',
    )


# ---------------------------------------------------------------------------
# Registry and CLI
# ---------------------------------------------------------------------------

_STANDALONE_RUN_SPECS: list[tuple[str, object, str]] = [
    ('e2e', run_vgg_unet_e2e_entry, 'vgg-unet-bh-e2e'),
]

_STANDALONE_VALID_SHORT_NAMES = frozenset(s[0] for s in _STANDALONE_RUN_SPECS)


def run_one(callback, wlname: str, cfg: dict):
    if IS_POLARIS:
        from ttsim.front.ttnn.device import close_device, open_device
        device = open_device()
    else:
        from ttnn import close_device, open_device  # type: ignore[no-redef]
        device = open_device(device_id=0, l1_small_size=32768, trace_region_size=6434816, num_command_queues=2)
    callback(wlname, device, cfg)
    close_device(device)


def standalone(test_name: str | None = None) -> None:
    """Run standalone BH e2e VGG UNet tests, or a single test by short name."""
    all_names = _STANDALONE_VALID_SHORT_NAMES | {'device-perf'}

    if test_name == 'device-perf':
        test_vgg_unet_perf_device()
        return

    if test_name is None:
        for _short, fn, wlname in _STANDALONE_RUN_SPECS:
            run_one(fn, wlname, {})
        return
    if test_name not in all_names:
        valid = ', '.join(sorted(all_names))
        logger.error(f'Unknown test {test_name}. Valid names: {valid}')
        sys.exit(1)
    for short, fn, wlname in _STANDALONE_RUN_SPECS:
        if short == test_name:
            run_one(fn, wlname, {})
            return


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stdout, level='INFO')
    parser = argparse.ArgumentParser(
        description='Run VGG UNet (Blackhole) e2e standalone tests.'
    )
    parser.add_argument(
        'test',
        nargs='?',
        metavar='TEST',
        default='e2e',
        help=(
            'Run only this test by short name, '
            'e.g. e2e, device-perf. If omitted, runs e2e.'
        ),
    )
    _args = parser.parse_args()
    standalone(_args.test)
