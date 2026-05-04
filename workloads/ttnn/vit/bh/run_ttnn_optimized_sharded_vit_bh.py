# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import sys

sys.path.append(".")

import os

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''

from loguru import logger  # noqa E402

if not IS_POLARIS:
    import pytest  # noqa E402
    import transformers  # type: ignore[import] # noqa E402
    import torch  # type: ignore[no-redef] # noqa E402
    import ttnn  # type: ignore[no-redef, import] # noqa E402
    from datasets import load_dataset  # type: ignore[import] # noqa F401, E402
    from transformers import AutoImageProcessor  # noqa E402
    from ttnn.model_preprocessing import preprocess_model_parameters  # type: ignore[import] # noqa F401, E402
else:
    import ttsim.front.ttnn as ttnn
    import ttsim.front.ttnn.minitorch_shim as torch  # type: ignore[no-redef]

    import workloads.ttnn.vit.bh.ttnn_optimized_sharded_vit_bh as ttnn_optimized_sharded_vit
    from ttsim.front.ttnn.device import set_default_device
    from ttsim.front.ttnn.tensor import ttnn_random
    torch_random = ttnn_random  # type: ignore[no-redef]

if not IS_POLARIS:
    from models.common.utility_functions import torch_random   # type: ignore[import, no-redef] # noqa F401, E402
    from models.demos.vision.classification.vit.common.common import load_torch_model   # type: ignore[import] # noqa F401, E402
    from models.demos.vision.classification.vit.blackhole.tt import (   # type: ignore[import] # noqa F401, E402
        ttnn_optimized_sharded_vit_bh as ttnn_optimized_sharded_vit,  # type: ignore[no-redef]
    )
    from tests.ttnn.utils_for_testing import assert_with_pcc   # type: ignore[import] # noqa F401, E402

if IS_POLARIS:
    from workloads.ttnn.vit.bh.vit_polaris_params_bh import (  # noqa: E402
        config_dict,
        config_obj,
        Parameters_attention_optimized,
        Parameters_dense_intermediate,
        Parameters_dense_output,
        _polaris_vit_encoder_layer_parameters,
        polaris_parameters_vit_patch_embeddings,
        polaris_parameters_vit_encoder,
        polaris_vit_parameters,
    )


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [10])
# @pytest.mark.parametrize("image_size", [224])
# @pytest.mark.parametrize("image_channels", [3])
def test_vit_patch_embeddings(
    device,
    model_name="google/vit-base-patch16-224",
    batch_size=10,
    image_size=224,
    image_channels=3,
    *,
    model_location_generator=None,
):
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    if not IS_POLARIS:
        model = load_torch_model(model_location_generator, embedding=True)
        config = model.config

        torch_pixel_values = torch_random(
            (batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.float32
        )
        torch_output, *_ = model.vit.embeddings.patch_embeddings(torch_pixel_values)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
        )

        torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
        torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
        batch_size, img_h, img_w, img_c = torch_pixel_values.shape
        patch_size = config.patch_size
        torch_pixel_values = torch_pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
        N, H, W, C = torch_pixel_values.shape
        shard_grid = ttnn.CoreRangeSet({
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 1)),
        })
        n_cores = 24
        shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)

        pixel_values = ttnn.from_torch(
            torch_pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                shard_spec,
            ),
        )
        config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)

        output = ttnn_optimized_sharded_vit.vit_patch_embeddings(
            config, pixel_values, parameters=parameters, unittest_check=True
        )
        output = ttnn.to_torch(output)
        assert_with_pcc(torch_output, output[0], 0.999)
    else:
        config = config_obj
        patch_size = config_dict["patch_size"]
        patch_count = image_size // patch_size
        patch_count_all = patch_count * patch_count
        hidden = config.hidden_size

        # Create pixel values already in the post-reshape shape [B, H, W/patch, 4*patch]
        # to avoid emitting host-side Permute/Pad/Reshape SimOps that don't exist on HW.
        pixel_values = ttnn.from_torch(
            torch_random(
                (batch_size, image_size, image_size // patch_size, 4 * patch_size),
                -1, 1, dtype=torch.bfloat16,
            ),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        parameters = polaris_parameters_vit_patch_embeddings()
        output = ttnn_optimized_sharded_vit.vit_patch_embeddings(
            config, pixel_values, parameters=parameters, unittest_check=True
        )
        output = ttnn.to_torch(output)
        expected_output_shape = [batch_size, patch_count_all, hidden]
        assert output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, but got {output.shape}"
        )
        logger.info(f"Obtained expected output shape {expected_output_shape}")


def run_vit_patch_embeddings(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit_patch_embeddings(device)


# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [10])
# @pytest.mark.parametrize("image_size", [224])
# @pytest.mark.parametrize("image_channels", [3])
def test_vit_embeddings(
    device,
    model_name="google/vit-base-patch16-224",
    batch_size=10,
    image_size=224,
    image_channels=3,
    *,
    model_location_generator=None,
):
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    if not IS_POLARIS:
        config = transformers.ViTConfig.from_pretrained(model_name)
        config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)
        model = load_torch_model(model_location_generator, embedding=True)

        dataset = load_dataset("huggingface/cats-image", revision="ccdec0af347ae11c5315146402c3e16c8bbf4149")
        image = dataset["test"]["image"][0]
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values
        torch_pixel_values = torch_pixel_values.repeat(batch_size, 1, 1, 1)
        torch_output, *_ = model.vit.embeddings(torch_pixel_values)

        model_state_dict = model.state_dict()
        torch_cls_token = model_state_dict["vit.embeddings.cls_token"]
        torch_position_embeddings = model_state_dict["vit.embeddings.position_embeddings"]
        if batch_size > 1:
            torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
        else:
            torch_cls_token = torch.nn.Parameter(torch_cls_token)
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)
        cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        position_embeddings = ttnn.from_torch(
            torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
        )

        torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
        torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
        batch_size, img_h, img_w, img_c = torch_pixel_values.shape
        patch_size = config.patch_size
        torch_pixel_values = torch_pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
        N, H, W, C = torch_pixel_values.shape
        shard_grid = ttnn.CoreRangeSet({
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 1)),
        })
        n_cores = 24
        shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)

        pixel_values = ttnn.from_torch(
            torch_pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                shard_spec,
            ),
        )

        output = ttnn_optimized_sharded_vit.vit_embeddings(
            config, pixel_values, cls_token, position_embeddings, parameters=parameters,
        )
        output = ttnn.to_torch(output)
        assert_with_pcc(torch_output, output[0][:197:], 0.999)
    else:
        config = config_obj
        patch_size = config_dict["patch_size"]
        sequence_len = 1 + (image_size // patch_size) ** 2
        hidden = config.hidden_size

        # Create pixel values already in the post-reshape shape [B, H, W/patch, 4*patch]
        # to avoid emitting host-side Permute/Pad/Reshape SimOps that don't exist on HW.
        pixel_values = ttnn.from_torch(
            torch_random(
                (batch_size, image_size, image_size // patch_size, 4 * patch_size),
                -1, 1, dtype=torch.bfloat16,
            ),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        torch_cls_token = torch_random((batch_size, 1, hidden), -0.1, 0.1, dtype=torch.bfloat16)
        torch_position_embeddings = torch_random(
            (batch_size, sequence_len, hidden), -0.1, 0.1, dtype=torch.bfloat16
        )
        cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        position_embeddings = ttnn.from_torch(torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

        parameters = polaris_parameters_vit_patch_embeddings()
        output = ttnn_optimized_sharded_vit.vit_embeddings(
            config, pixel_values, cls_token, position_embeddings, parameters=parameters,
        )
        output = ttnn.to_torch(output)
        expected_output_shape = [batch_size, sequence_len, hidden]
        assert output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, but got {output.shape}"
        )
        logger.info(f"Obtained expected output shape {expected_output_shape}")


def run_vit_embeddings(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit_embeddings(device)


# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [10])
# @pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_attention(device, model_name="google/vit-base-patch16-224",
                       batch_size=10, sequence_size=224):
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    if not IS_POLARIS:
        config = transformers.ViTConfig.from_pretrained(model_name)
        config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)
        model = transformers.models.vit.modeling_vit.ViTAttention(config).eval()

        torch_hidden_states = torch_random(
            (batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32
        )
        torch_output, *_ = model(torch_hidden_states)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
        )

        hidden_states = ttnn.from_torch(
            torch_hidden_states,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        encoder_input = ttnn.to_memory_config(
            hidden_states,
            memory_config=ttnn.create_sharded_memory_config(
                hidden_states.shape,
                core_grid=config.core_grid,  # BH dynamic core_grid
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(hidden_states)

        output = ttnn_optimized_sharded_vit.vit_attention(
            config, encoder_input, parameters=parameters,
        )
        output = ttnn.to_torch(output)
        assert_with_pcc(torch_output, output, 0.999)
    else:
        config = config_obj
        torch_hidden_states = torch_random(
            (batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16
        )
        hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        parameters = Parameters_attention_optimized()

        output = ttnn_optimized_sharded_vit.vit_attention(
            config, hidden_states, parameters=parameters,
        )
        output = ttnn.to_torch(output)
        expected_output_shape = [batch_size, sequence_size, config.hidden_size]
        assert output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, but got {output.shape}"
        )
        logger.info(f"Obtained expected output shape {expected_output_shape}")


def run_vit_attention(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit_attention(device)


# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [10])
# @pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_intermediate(device, model_name="google/vit-base-patch16-224",
                          batch_size=10, sequence_size=224):
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    if not IS_POLARIS:
        config = transformers.ViTConfig.from_pretrained(model_name)
        config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)
        model = transformers.models.vit.modeling_vit.ViTIntermediate(config).eval()

        torch_hidden_states = torch_random(
            (batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32
        )
        torch_output = model(torch_hidden_states)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model.to(torch.bfloat16),
            device=device,
        )

        hidden_states = ttnn.from_torch(
            torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

        output = ttnn_optimized_sharded_vit.vit_intermediate(
            config, hidden_states, parameters=parameters,
        )
        output = ttnn.to_torch(output)
        assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9984)
    else:
        config = config_obj
        torch_hidden_states = torch_random(
            (batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16
        )
        hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        parameters = Parameters_dense_intermediate()

        output = ttnn_optimized_sharded_vit.vit_intermediate(
            config, hidden_states, parameters=parameters,
        )
        output = ttnn.to_torch(output)
        expected_output_shape = [batch_size, sequence_size, config.intermediate_size]
        assert output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, but got {output.shape}"
        )
        logger.info(f"Obtained expected output shape {expected_output_shape}")


def run_vit_intermediate(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit_intermediate(device)


# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [10])
# @pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_output(device, model_name="google/vit-base-patch16-224",
                    batch_size=10, sequence_size=224):
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    if not IS_POLARIS:
        config = transformers.ViTConfig.from_pretrained(model_name)
        config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)
        model = transformers.models.vit.modeling_vit.ViTOutput(config).eval()

        torch_intermediate = torch_random(
            (batch_size, sequence_size, config.intermediate_size), -1, 1, dtype=torch.float32
        )
        torch_residual = torch_random(
            (batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32
        )
        torch_output = model(torch_intermediate, torch_residual)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
        )

        intermediate = ttnn.from_torch(
            torch_intermediate, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )
        residual = ttnn.from_torch(
            torch_residual, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

        residual_sh = ttnn.to_memory_config(
            residual,
            memory_config=ttnn.create_sharded_memory_config(
                residual.shape,
                core_grid=config.core_grid,  # BH dynamic core_grid
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(residual)

        output = ttnn_optimized_sharded_vit.vit_output(
            config, intermediate, residual_sh, parameters=parameters,
        )
        output = ttnn.to_torch(output)
        assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.999)
    else:
        config = config_obj
        torch_intermediate = torch_random(
            (batch_size, sequence_size, config.intermediate_size), -0.1, 0.1, dtype=torch.bfloat16
        )
        intermediate = ttnn.from_torch(torch_intermediate, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        residual = ttnn.from_torch(
            torch_random(
                (batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16
            ),
            dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
        )
        parameters = Parameters_dense_output()

        output = ttnn_optimized_sharded_vit.vit_output(
            config, intermediate, residual, parameters=parameters,
        )
        output = ttnn.to_torch(output)
        expected_output_shape = [batch_size, sequence_size, config.hidden_size]
        assert output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, but got {output.shape}"
        )
        logger.info(f"Obtained expected output shape {expected_output_shape}")


def run_vit_output(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit_output(device)


# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [10])
# @pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_layer(
    device,
    model_name="google/vit-base-patch16-224",
    batch_size=10,
    sequence_size=224,
    *,
    model_location_generator=None,
):
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    if not IS_POLARIS:
        config = transformers.ViTConfig.from_pretrained(model_name)
        config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)
        model = load_torch_model(model_location_generator, embedding=True).vit.encoder.layer[0]

        torch_hidden_states = torch_random(
            (batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32
        )
        torch_output, *_ = model(torch_hidden_states)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
            device=device,
        )

        hidden_states = ttnn.from_torch(
            torch_hidden_states,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        encoder_input = ttnn.to_memory_config(
            hidden_states,
            memory_config=ttnn.create_sharded_memory_config(
                hidden_states.shape,
                core_grid=config.core_grid,  # BH dynamic core_grid
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(hidden_states)

        output = ttnn_optimized_sharded_vit.vit_layer(
            config, encoder_input, parameters=parameters,
        )
        output = ttnn.to_torch(output)
        assert_with_pcc(torch_output, output, 0.985)
    else:
        config = config_obj
        torch_hidden_states = torch_random(
            (batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.bfloat16
        )
        hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        parameters = _polaris_vit_encoder_layer_parameters()

        output = ttnn_optimized_sharded_vit.vit_layer(
            config, hidden_states, parameters=parameters,
        )
        output = ttnn.to_torch(output)
        expected_output_shape = [batch_size, sequence_size, config.hidden_size]
        assert output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, but got {output.shape}"
        )
        logger.info(f"Obtained expected output shape {expected_output_shape}")


def run_vit_layer(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit_layer(device)


# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [10])
# @pytest.mark.parametrize("sequence_size", [224])  # padded from 197 to 224
def test_vit_encoder(
    device,
    model_name="google/vit-base-patch16-224",
    batch_size=10,
    sequence_size=224,
    *,
    model_location_generator=None,
):
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    if not IS_POLARIS:
        model = load_torch_model(model_location_generator, embedding=True)
        config = model.config
        model = model.vit.encoder
        model = model.to(torch.float32)
        config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)

        torch_hidden_states = torch_random(
            (batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32
        )
        torch_output = model(torch_hidden_states).last_hidden_state

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
            device=device,
        )

        hidden_states = ttnn.from_torch(
            torch_hidden_states,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        output = ttnn_optimized_sharded_vit.vit_encoder(
            config, hidden_states, parameters=parameters,
        )
        output = ttnn.to_torch(output)
        assert_with_pcc(torch_output, output, 0.96)
    else:
        config = config_obj
        torch_hidden_states = torch_random(
            (batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16
        )
        hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        parameters = polaris_parameters_vit_encoder()

        output = ttnn_optimized_sharded_vit.vit_encoder(
            config, hidden_states, parameters=parameters,
        )
        output = ttnn.to_torch(output)
        expected_output_shape = [batch_size, sequence_size, config.hidden_size]
        assert output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, but got {output.shape}"
        )
        logger.info(f"Obtained expected output shape {expected_output_shape}")


def run_vit_encoder(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit_encoder(device)


# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [10])
# @pytest.mark.parametrize("image_size", [224])
# @pytest.mark.parametrize("image_channels", [3])
# @pytest.mark.parametrize("sequence_size", [224])
def test_vit(
    device,
    model_name="google/vit-base-patch16-224",
    batch_size=10,
    image_size=224,
    image_channels=3,
    sequence_size=224,
    *,
    model_location_generator=None,
):
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    if not IS_POLARIS:
        model = load_torch_model(model_location_generator, embedding=True)
        config = model.config
        config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)

        dataset = load_dataset("huggingface/cats-image", revision="ccdec0af347ae11c5315146402c3e16c8bbf4149")
        image = dataset["test"]["image"][0]
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values
        torch_pixel_values = torch_pixel_values.repeat(batch_size, 1, 1, 1)
        torch_output, *_ = model(torch_pixel_values).logits

        model_state_dict = model.state_dict()
        torch_cls_token = model_state_dict["vit.embeddings.cls_token"]
        torch_position_embeddings = model_state_dict["vit.embeddings.position_embeddings"]
        if batch_size > 1:
            torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
        else:
            torch_cls_token = torch.nn.Parameter(torch_cls_token)
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)
        cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        position_embeddings = ttnn.from_torch(
            torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
        )

        torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
        torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
        batch_size, img_h, img_w, img_c = torch_pixel_values.shape
        patch_size = config.patch_size
        torch_pixel_values = torch_pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
        N, H, W, C = torch_pixel_values.shape

        if batch_size <= 10:
            fold_core_x = batch_size - 1
            fold_core_y = 1
        else:
            batch_size = 20  # Use multiple of 10 for optimal blackhole utilization
            fold_core_x = 11  # Use full x-dimension (0-11 = 12 cores)
            fold_core_y = 1   # Use 2 rows (0-1 = 2 rows) for 12x2=24 cores total

        shard_grid = ttnn.CoreRangeSet({
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(fold_core_x, fold_core_y)),
        })
        n_cores = batch_size * 2
        shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)

        pixel_values = ttnn.from_torch(
            torch_pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                shard_spec,
            ),
        )

        output = ttnn_optimized_sharded_vit.vit(
            config, pixel_values, cls_token, position_embeddings, parameters=parameters,
        )
        output = ttnn.to_torch(output)
        assert_with_pcc(torch_output, output[0, 0, :1000], 0.879)
    else:
        config = config_obj
        patch_size = config_dict["patch_size"]
        num_labels = 1152
        sequence_len = 1 + (image_size // patch_size) ** 2

        # Create pixel values already in the post-reshape shape [B, H, W/patch, 4*patch]
        # to avoid emitting host-side Permute/Pad/Reshape SimOps that don't exist on HW.
        pixel_values = ttnn.from_torch(
            torch_random(
                (batch_size, image_size, image_size // patch_size, 4 * patch_size),
                -1, 1, dtype=torch.bfloat16,
            ),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        torch_cls_token = torch_random((batch_size, 1, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
        torch_position_embeddings = torch_random(
            (batch_size, sequence_len, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16
        )
        cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        position_embeddings = ttnn.from_torch(torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

        parameters = polaris_vit_parameters(num_labels=num_labels)
        output = ttnn_optimized_sharded_vit.vit(
            config, pixel_values, cls_token, position_embeddings, parameters=parameters,
        )
        output = ttnn.to_torch(output)
        expected_output_shape = [batch_size, sequence_len, num_labels]
        assert output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, but got {output.shape}"
        )
        logger.info(f"Obtained expected output shape {expected_output_shape}")


def run_vit(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit(device)


# ---------------------------------------------------------------------------
# Registry and CLI
# ---------------------------------------------------------------------------

_STANDALONE_RUN_SPECS: list[tuple[str, object, str]] = [
    ("patch-embeddings", run_vit_patch_embeddings, "opt-vit-bh-patch-embeddings"),
    ("embeddings", run_vit_embeddings, "opt-vit-bh-embeddings"),
    ("attention", run_vit_attention, "opt-vit-bh-attention"),
    ("intermediate", run_vit_intermediate, "opt-vit-bh-intermediate"),
    ("output", run_vit_output, "opt-vit-bh-output"),
    ("layer", run_vit_layer, "opt-vit-bh-layer"),
    ("encoder", run_vit_encoder, "opt-vit-bh-encoder"),
    ("vit", run_vit, "opt-vit-bh"),
]

_STANDALONE_VALID_SHORT_NAMES = frozenset(s[0] for s in _STANDALONE_RUN_SPECS)


def run_one(callback, wlname: str, cfg: dict):
    if IS_POLARIS:
        from ttsim.front.ttnn.device import close_device, open_device
        device = open_device()
    else:
        from ttnn import close_device, open_device
        device = open_device(device_id=0)
    callback(wlname, device, cfg)
    close_device(device)


def standalone(test_name: str | None = None) -> None:
    """Run all standalone optimized sharded ViT (BH) tests, or a single test by short name."""
    if test_name is None:
        for _short, fn, wlname in _STANDALONE_RUN_SPECS:
            run_one(fn, wlname, {})
        return
    if test_name not in _STANDALONE_VALID_SHORT_NAMES:
        valid = ", ".join(sorted(_STANDALONE_VALID_SHORT_NAMES))
        logger.error(
            f"Unknown test {test_name}. Valid names: {valid}"
        )
        sys.exit(1)
    for short, fn, wlname in _STANDALONE_RUN_SPECS:
        if short == test_name:
            run_one(fn, wlname, {})
            return


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    parser = argparse.ArgumentParser(
        description="Run optimized sharded ViT (Blackhole) standalone tests."
    )
    parser.add_argument(
        "test",
        nargs="?",
        metavar="TEST",
        default="attention",
        help=(
            "Run only this test by short name, "
            "e.g. attention, patch-embeddings, vit. If omitted, runs 'attention'."
        ),
    )
    _args = parser.parse_args()
    standalone(_args.test)
