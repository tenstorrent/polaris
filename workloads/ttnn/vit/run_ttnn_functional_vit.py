# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
    import ttnn
    import torch
    from ttnn.model_preprocessing import preprocess_model_parameters  # type: ignore[import] # noqa F401,  E402
else:
    import ttsim.front.ttnn as ttnn
    import ttsim.front.ttnn.minitorch_shim as torch  # type: ignore[no-redef]
    import types

    import workloads.ttnn.vit.ttnn_functional_vit as ttnn_functional_vit
    from ttsim.front.ttnn.device import set_default_device
    from ttsim.front.ttnn.tensor import ttnn_random
    from ttsim.front.ttnn.ttnn_shim import permute_op
    torch_random = ttnn_random  # type: ignore[no-redef]
    # torch_bfloat16 = ttnn.DataType.BFLOAT16

if not IS_POLARIS:
    from models.common.utility_functions import is_blackhole, is_wormhole_b0, torch_random   # type: ignore[import, no-redef] # noqa F401, E402
    from models.demos.vision.classification.vit.common.common import load_torch_model   # type: ignore[import] # noqa F401, E402
    from models.demos.vision.classification.vit.common.tt import ttnn_functional_vit   # type: ignore[no-redef, import] # noqa F401, E402

    from tests.ttnn.utils_for_testing import assert_with_pcc   # type: ignore[import] # noqa F401, E402

if IS_POLARIS:
    config_dict = {
      "architectures": ["ViTForImageClassification"  ],
      "attention_probs_dropout_prob": 0.0,
      "encoder_stride": 16,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.0,
      "hidden_size": 768,
      "id2label": {  },
      "image_size": 224,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "label2id": {  },
      "layer_norm_eps": 1e-12,
      "model_type": "vit",
      "num_attention_heads": 12,
      "num_channels": 3,
      "num_hidden_layers": 12,
      "patch_size": 16,
      "pooler_act": "tanh",
      "pooler_output_size": 768,
      "qkv_bias": True,
      "transformers_version": "4.53.0"
    }

    config_obj = types.SimpleNamespace(**config_dict)


    def make_info(weight_shape, bias_shape):
        return types.SimpleNamespace({
            "weight": ttnn.Tensor(shape=weight_shape, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT),
            "bias": ttnn.Tensor(shape=bias_shape, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT)
        })

    def _polaris_vit_embeddings_patch_parameters():
        """Embeddings.patch_embeddings subtree (projection only) for vit_patch_embeddings unittest_check."""
        hidden = config_dict["hidden_size"]
        return types.SimpleNamespace(
            patch_embeddings=types.SimpleNamespace(
                projection=make_info(
                    weight_shape=ttnn.Shape([1024, hidden]),
                    bias_shape=ttnn.Shape([1, hidden]),
                )
            )
        )

    def polaris_parameters_vit_patch_embeddings():
        """Full parameters root for vit_patch_embeddings(..., unittest_check=True)."""
        return types.SimpleNamespace(
            vit=types.SimpleNamespace(embeddings=_polaris_vit_embeddings_patch_parameters())
        )

    class Parameters_qkv:
        def __init__(self):
            query = make_info(weight_shape=ttnn.Shape([768, 768]), bias_shape=ttnn.Shape([1, 768]))
            key = make_info(weight_shape=ttnn.Shape([768, 768]), bias_shape=ttnn.Shape([1, 768]))
            value = make_info(weight_shape=ttnn.Shape([768, 768]), bias_shape=ttnn.Shape([1, 768]))
            dense = make_info(weight_shape=ttnn.Shape([768, 768]), bias_shape=ttnn.Shape([1, 768]))
            self.dense = dense
            self.output = types.SimpleNamespace(dense=dense)
            self.attention = types.SimpleNamespace(query=query, key=key, value=value, dense=dense, output=self.output)

    class Parameters_dense_intermediate:
        def __init__(self):
            dense = make_info(weight_shape=ttnn.Shape([768, 3072]), bias_shape=ttnn.Shape([1, 3072]))
            self.dense = dense
    class Parameters_dense_output:
        def __init__(self):
            dense = make_info(weight_shape=ttnn.Shape([3072, 768]), bias_shape=ttnn.Shape([1, 768]))
            self.dense = dense

    def _polaris_vit_encoder_layer_parameters():
        """Single ViT encoder block parameters for vit_layer / vit_encoder.layer[i]."""
        hidden = config_dict["hidden_size"]
        qkv = Parameters_qkv()
        return types.SimpleNamespace(
            layernorm_before=make_info(
                weight_shape=ttnn.Shape([1, hidden]),
                bias_shape=ttnn.Shape([1, hidden]),
            ),
            layernorm_after=make_info(
                weight_shape=ttnn.Shape([1, hidden]),
                bias_shape=ttnn.Shape([1, hidden]),
            ),
            attention=qkv.attention,
            intermediate=Parameters_dense_intermediate(),
            output=Parameters_dense_output(),
        )

    def polaris_parameters_vit_encoder():
        """Encoder-only parameters for vit_encoder (stack of transformer blocks)."""
        num_layers = config_dict["num_hidden_layers"]
        return types.SimpleNamespace(
            layer=[_polaris_vit_encoder_layer_parameters() for _ in range(num_layers)]
        )

    def polaris_vit_parameters(*, num_labels: int = 1000):
        """Full ViT parameter tree for POLARIS (no torch model). Shapes match ViT-Base / ttnn matmul paths."""
        hidden = config_dict["hidden_size"]
        image_size = config_dict["image_size"]
        patch_size = config_dict["patch_size"]
        patches_per_side = image_size // patch_size
        num_patch_tokens = patches_per_side * patches_per_side
        num_cls_tokens = 1
        sequence_length = num_patch_tokens + num_cls_tokens  # noqa: F841 — token dim incl. CLS (e.g. 197 for ViT-B/224)

        embeddings = _polaris_vit_embeddings_patch_parameters()

        encoder = polaris_parameters_vit_encoder()
        layernorm = make_info(
            weight_shape=ttnn.Shape([1, hidden]),
            bias_shape=ttnn.Shape([1, hidden]),
        )
        vit_ns = types.SimpleNamespace(
            embeddings=embeddings,
            encoder=encoder,
            layernorm=layernorm,
        )
        classifier = make_info(
            weight_shape=ttnn.Shape([hidden, num_labels]),
            bias_shape=ttnn.Shape([1, num_labels]),
        )
        return types.SimpleNamespace(vit=vit_ns, classifier=classifier)

# @pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
# @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [8])
# @pytest.mark.parametrize("image_size", [224])
# @pytest.mark.parametrize("image_channels", [3])
def test_vit_patch_embeddings(
    device,
    model_name="google/vit-base-patch16-224",
    batch_size=8,
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
        model = model.to(torch.bfloat16)
        config = model.config

        torch_pixel_values = torch_random(
            (batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.bfloat16
        )
        torch_output, *_ = model(torch_pixel_values)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_functional_vit.custom_preprocessor,
        )

        pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
        pixel_values = torch.nn.functional.pad(pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
        pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        output = ttnn_functional_vit.vit_patch_embeddings(
            config, pixel_values, parameters=parameters, unittest_check=True
        )
        output = ttnn.to_torch(output)

        torch_output, *_ = model.vit.embeddings.patch_embeddings(torch_pixel_values)
        assert_with_pcc(torch_output, output[0], 0.9999)
    else:
        config = config_obj  # type: ignore[assignment]
        patch_size = config_dict["patch_size"]
        patch_count = image_size // patch_size
        patch_count_all = patch_count * patch_count
        hidden = config.hidden_size

        torch_pixel_values = torch_random(
            (batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.bfloat16
        )
        pixel_values = permute_op(torch_pixel_values, [0, 2, 3, 1])
        pixel_values = ttnn.pad(pixel_values, [0, 1, 0, 0, 0, 0, 0, 0], value=0)
        pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        parameters = polaris_parameters_vit_patch_embeddings()
        output = ttnn_functional_vit.vit_patch_embeddings(
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

# @pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
# @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [8])
# @pytest.mark.parametrize("image_size", [224])
# @pytest.mark.parametrize("image_channels", [3])
def test_vit_embeddings(
    device,
    model_name="google/vit-base-patch16-224",
    batch_size=8,
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
        model = model.to(torch.bfloat16)
        config = model.config

        dataset = load_dataset("huggingface/cats-image")
        image = dataset["test"]["image"][0]
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(torch.bfloat16)
        torch_pixel_values = torch_pixel_values.repeat(batch_size, 1, 1, 1)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_functional_vit.custom_preprocessor,
        )

        # cls_token & position embeddings expand to batch_size
        # TODO: pass batch_size to preprocess_model_parameters
        model_state_dict = model.state_dict()
        torch_cls_token = model_state_dict["vit.embeddings.cls_token"]
        torch_position_embeddings = model_state_dict["vit.embeddings.position_embeddings"]
        if batch_size > 1:
            torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
        else:
            torch_cls_token = torch.nn.Parameter(torch_cls_token)
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)
        cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        position_embeddings = ttnn.from_torch(
            torch_position_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
        pixel_values = torch.nn.functional.pad(pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
        pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        output = ttnn_functional_vit.vit_embeddings(
            config,
            pixel_values,
            cls_token,
            position_embeddings,
            parameters=parameters,
        )
        output = ttnn.to_torch(output)

        torch_output, *_ = model.vit.embeddings(torch_pixel_values)

        assert_with_pcc(torch_output, output[0], 0.9999)
    else:
        config = config_obj  # type: ignore[assignment]
        patch_size = config_dict["patch_size"]
        sequence_len = 1 + (image_size // patch_size) ** 2
        hidden = config.hidden_size

        torch_pixel_values = torch_random(
            (batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.bfloat16
        )
        pixel_values = permute_op(torch_pixel_values, [0, 2, 3, 1])
        pixel_values = ttnn.pad(pixel_values, [0, 1, 0, 0, 0, 0, 0, 0], value=0)
        pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        torch_cls_token = torch_random((batch_size, 1, hidden), -0.1, 0.1, dtype=torch.bfloat16)
        torch_position_embeddings = torch_random(
            (batch_size, sequence_len, hidden), -0.1, 0.1, dtype=torch.bfloat16
        )
        cls_token = ttnn.from_torch(torch_cls_token, layout=ttnn.TILE_LAYOUT)
        position_embeddings = ttnn.from_torch(torch_position_embeddings, layout=ttnn.TILE_LAYOUT)

        parameters = polaris_parameters_vit_patch_embeddings()
        output = ttnn_functional_vit.vit_embeddings(
            config,
            pixel_values,
            cls_token,
            position_embeddings,
            parameters=parameters,
        )
        output = ttnn.to_torch(output)
        expected_output_shape = [batch_size, sequence_len, hidden]
        assert output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, but got {output.shape}"
        )
        logger.info(f"Obtained expected output shape {expected_output_shape}")


def run_vit_embeddings(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit_embeddings(device)


# @pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
# @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [8])
# @pytest.mark.parametrize("sequence_size", [224])
def test_vit_attention(device, model_name="google/vit-base-patch16-224",
                       batch_size=8, sequence_size=224):
    assert isinstance(model_name, str)
    assert isinstance(batch_size, int)
    assert isinstance(sequence_size, int)
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    if not IS_POLARIS:
        config = transformers.ViTConfig.from_pretrained(model_name)
        model = transformers.models.vit.modeling_vit.ViTAttention(config).eval()
        model = model.to(torch.bfloat16)
        torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
        torch_attention_mask = torch.ones(1, sequence_size, dtype=torch.bfloat16)  # type: ignore[attr-defined]
        torch_output, *_ = model(torch_hidden_states, torch_attention_mask)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_functional_vit.custom_preprocessor,  # type: ignore[attr-defined]
        )

        hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
        attention_mask = ttnn.from_torch(torch_attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        config = config_obj  # type: ignore[assignment]
        # NOTE: .to(ttnn.TILE_LAYOUT) is a no-op for ttsim Tensor; only affects host tensors.
        # NOTE: HW code uses from_torch to create tensors in TILE_LAYOUT.
        # NOTE: Without TILE_LAYOUT here, the untilize ops will not be used
        torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
        hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT)
        attention_mask = ttnn.ones(1, sequence_size, dtype=torch.bfloat16, layout=ttnn.TILE_LAYOUT)
        parameters = Parameters_qkv()
    output = ttnn_functional_vit.vit_attention(
        config,
        hidden_states,
        attention_mask=attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)
    if IS_POLARIS:
        expected_output_shape = [batch_size, sequence_size, config.hidden_size]
        assert output.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, but got {output.shape}"
        logger.info(f"Obtained expected output shape {expected_output_shape}")
    else:
        assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9999)

def run_vit_attention(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit_attention(device)

# @pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
# @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [8])
# @pytest.mark.parametrize("sequence_size", [224])
# @pytest.mark.parametrize("torch_dtype", [torch.bfloat16])
def test_vit_intermediate(device,
                          model_name = "google/vit-base-patch16-224",
                          batch_size=8,
                          sequence_size=224,
                          torch_dtype=torch.bfloat16):
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    if not IS_POLARIS:
        config = transformers.ViTConfig.from_pretrained(model_name)
        model = transformers.models.vit.modeling_vit.ViTIntermediate(config).eval()
        model = model.to(torch.bfloat16)

        torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
        torch_output = model(torch_hidden_states)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_functional_vit.custom_preprocessor,  # type: ignore[attr-defined]
        )

        hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT, device=device)

    else:
        config = config_obj  # type: ignore[assignment]
        torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
        hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT)
        parameters = Parameters_dense_intermediate()


    output = ttnn_functional_vit.vit_intermediate(
        hidden_states,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)
    if IS_POLARIS:
        expected_output_shape = [batch_size, sequence_size, config.intermediate_size]
        assert output.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, but got {output.shape}"
    else:
        assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9999)


def run_vit_intermediate(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit_intermediate(device)


# @pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
# @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [8])
# @pytest.mark.parametrize("sequence_size", [224])
def test_vit_output(device,
                    model_name="google/vit-base-patch16-224",
                    batch_size=8,
                    sequence_size=224):
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    if not IS_POLARIS:
        config = transformers.ViTConfig.from_pretrained(model_name)
        model = transformers.models.vit.modeling_vit.ViTOutput(config).eval()
        model = model.to(torch.bfloat16)

        torch_intermediate = torch_random(
            (batch_size, sequence_size, config.intermediate_size), -0.1, 0.1, dtype=torch.bfloat16
        )
        torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
        torch_output = model(torch_intermediate, torch_residual)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_functional_vit.custom_preprocessor,  # type: ignore[attr-defined]
        )

        intermediate = ttnn.from_torch(torch_intermediate, layout=ttnn.TILE_LAYOUT, device=device)
        residual = ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device)

    else:
        config = config_obj  # type: ignore[assignment]
        torch_intermediate = torch_random(
            (batch_size, sequence_size, config.intermediate_size), -0.1, 0.1, dtype=torch.bfloat16
        )
        intermediate = ttnn.from_torch(torch_intermediate, layout=ttnn.TILE_LAYOUT)
        residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
        parameters = Parameters_dense_output()
    output = ttnn_functional_vit.vit_output(
        config,
        intermediate,
        residual,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    if IS_POLARIS:
        expected_output_shape = [batch_size, sequence_size, config.hidden_size]
        assert output.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, but got {output.shape}"
    else:
        assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9999)  # 9994

def run_vit_output(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit_output(device)


# @pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
# @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [8])
# @pytest.mark.parametrize("sequence_size", [224])
def test_vit_layer(
    device,
    model_name="google/vit-base-patch16-224",
    batch_size=8,
    sequence_size=224,
    *,
    model_location_generator=None,
):
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    if not IS_POLARIS:

        model = load_torch_model(model_location_generator, embedding=True)
        model = model.to(torch.bfloat16)
        config = model.config
        model = model.vit.encoder.layer[0]

        torch_hidden_states = torch_random(
            (batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.bfloat16
        )
        torch_attention_mask = torch.ones(1, sequence_size, dtype=torch.bfloat16)
        torch_output, *_ = model(torch_hidden_states, torch_attention_mask)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_functional_vit.custom_preprocessor,
        )

        hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
        attention_mask = ttnn.from_torch(torch_attention_mask, layout=ttnn.TILE_LAYOUT, device=device)

        output = ttnn_functional_vit.vit_layer(
            config,
            hidden_states,
            attention_mask=attention_mask,
            parameters=parameters,
        )
        output = ttnn.to_torch(output)

        assert_with_pcc(torch_output, output, 0.9999)  # 0.9957
    else:
        config = config_obj  # type: ignore[assignment]
        torch_hidden_states = torch_random(
            (batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.bfloat16
        )
        hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT)
        attention_mask = ttnn.ones(1, sequence_size, dtype=torch.bfloat16, layout=ttnn.TILE_LAYOUT)
        parameters = _polaris_vit_encoder_layer_parameters()
        output = ttnn_functional_vit.vit_layer(
            config,
            hidden_states,
            attention_mask=attention_mask,
            parameters=parameters,
        )
        output = ttnn.to_torch(output)
        expected_output_shape = [batch_size, sequence_size, config.hidden_size]
        assert output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, but got {output.shape}"
        )
        logger.info(f"Obtained expected output shape {expected_output_shape}")


def run_vit_layer(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit_layer(device)

# @pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
# @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [8])
# @pytest.mark.parametrize("sequence_size", [224])
def test_vit_encoder(
    device,
    model_name="google/vit-base-patch16-224",
    batch_size=8,
    sequence_size=224,
    *,
    model_location_generator=None,
):
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    if not IS_POLARIS:

        model = load_torch_model(model_location_generator)
        model = model.to(torch.bfloat16)
        config = model.config
        model = model.vit.encoder

        torch_hidden_states = torch_random(
            (batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16
        )
        torch_attention_mask = None
        torch_output = model(torch_hidden_states, torch_attention_mask).last_hidden_state

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_functional_vit.custom_preprocessor,
        )

        hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
        if torch_attention_mask is not None:
            attention_mask = ttnn.from_torch(
                torch_attention_mask, layout=ttnn.TILE_LAYOUT, device=device
            )
        else:
            attention_mask = None

        output = ttnn_functional_vit.vit_encoder(
            config,
            hidden_states,
            attention_mask=attention_mask,
            parameters=parameters,
        )
        output = ttnn.to_torch(output)

        assert_with_pcc(torch_output, output, 0.9999)  # 0.9294
    else:
        config = config_obj  # type: ignore[assignment]
        torch_hidden_states = torch_random(
            (batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16
        )
        hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        attention_mask = None
        parameters = polaris_parameters_vit_encoder()
        output = ttnn_functional_vit.vit_encoder(
            config,
            hidden_states,
            attention_mask=attention_mask,
            parameters=parameters,
        )
        output = ttnn.to_torch(output)
        expected_output_shape = [batch_size, sequence_size, config.hidden_size]
        assert output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, but got {output.shape}"
        )
        logger.info(f"Obtained expected output shape {expected_output_shape}")


def run_vit_encoder(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit_encoder(device)


# @pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
# @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
# @pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
# @pytest.mark.parametrize("batch_size", [8])
# @pytest.mark.parametrize("image_size", [224])
# @pytest.mark.parametrize("image_channels", [3])
def test_vit(
    device,
    model_name="google/vit-base-patch16-224",
    batch_size=8,
    image_size=224,
    image_channels=3,
    *,
    model_location_generator=None,
):
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    if not IS_POLARIS:

        model = load_torch_model(model_location_generator)
        model = model.to(torch.bfloat16)
        config = model.config

        dataset = load_dataset("huggingface/cats-image")
        image = dataset["train"][0]["image"]
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(torch.bfloat16)
        torch_pixel_values = torch_pixel_values.repeat(batch_size, 1, 1, 1)

        torch_output, *_ = model(torch_pixel_values).logits

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_functional_vit.custom_preprocessor,
        )

        # cls_token & position embeddings expand to batch_size
        # TODO: pass batch_size to preprocess_model_parameters
        model_state_dict = model.state_dict()
        torch_cls_token = model_state_dict["vit.embeddings.cls_token"]
        torch_position_embeddings = model_state_dict["vit.embeddings.position_embeddings"]
        if batch_size > 1:
            torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
        else:
            torch_cls_token = torch.nn.Parameter(torch_cls_token)
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)
        cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        position_embeddings = ttnn.from_torch(
            torch_position_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
        pixel_values = torch.nn.functional.pad(pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
        pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        output = ttnn_functional_vit.vit(
            config,
            pixel_values,
            None,
            cls_token,
            position_embeddings,
            parameters=parameters,
        )
        output = ttnn.to_torch(output)

        assert_with_pcc(torch_output, output[0][0], 0.9996)  # 0.9806
    else:
        config = config_obj  # type: ignore[assignment]
        num_labels = 1000
        sequence_len = 1 + (image_size // config_dict["patch_size"]) ** 2

        torch_pixel_values = torch_random(
            (batch_size, image_channels, image_size, image_size), -1, 1, dtype=torch.bfloat16
        )
        # minitorch_shim has no permute / F.pad; use ttsim ops on tensors (NCHW→NHWC + pad 3→4 ch).
        pixel_values = permute_op(torch_pixel_values, [0, 2, 3, 1])
        pixel_values = ttnn.pad(pixel_values, [0, 1, 0, 0, 0, 0, 0, 0], value=0)
        pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        torch_cls_token = torch_random((batch_size, 1, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
        torch_position_embeddings = torch_random(
            (batch_size, sequence_len, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16
        )
        cls_token = ttnn.from_torch(torch_cls_token, layout=ttnn.TILE_LAYOUT)
        position_embeddings = ttnn.from_torch(torch_position_embeddings, layout=ttnn.TILE_LAYOUT)

        parameters = polaris_vit_parameters(num_labels=num_labels)
        output = ttnn_functional_vit.vit(
            config,
            pixel_values,
            None,
            cls_token,
            position_embeddings,
            parameters=parameters,
        )
        output = ttnn.to_torch(output)
        expected_output_shape = [batch_size, sequence_len, num_labels]
        assert output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, but got {output.shape}"
        )
        logger.info(f"Obtained expected output shape {expected_output_shape}")


def run_vit(wlname: str, device: ttnn.device.Device, cfg: dict):
    return test_vit(device)


# Short names omit the "vit-" workload prefix (full-model workload is just "vit" → short name "vit").
_STANDALONE_RUN_SPECS: list[tuple[str, object, str]] = [
    ("attention", run_vit_attention, "vit-attention"),
    ("vit", run_vit, "vit"),
    ("patch-embeddings", run_vit_patch_embeddings, "vit-patch-embeddings"),
    ("intermediate", run_vit_intermediate, "vit-intermediate"),
    ("output", run_vit_output, "vit-output"),
    ("layer", run_vit_layer, "vit-layer"),
    ("encoder", run_vit_encoder, "vit-encoder"),
    ("embeddings", run_vit_embeddings, "vit-embeddings"),
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
    """Run all standalone ViT tests, or a single test selected by short name (no ``vit-`` prefix)."""
    if test_name is None:
        for _short, fn, wlname in _STANDALONE_RUN_SPECS:
            run_one(fn, wlname, {})
        return
    if test_name not in _STANDALONE_VALID_SHORT_NAMES:
        valid = ", ".join(sorted(_STANDALONE_VALID_SHORT_NAMES))
        logger.error(
            f"Unknown test {test_name}. Valid names (without vit- prefix): {valid}"
        )
        sys.exit(1)
    for short, fn, wlname in _STANDALONE_RUN_SPECS:
        if short == test_name:
            run_one(fn, wlname, {})
            return


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    parser = argparse.ArgumentParser(description="Run ViT functional standalone tests.")
    parser.add_argument(
        "test",
        nargs="?",
        metavar="TEST",
        default="patch-embeddings",
        help=(
            "Run only this test by short name (omit the vit- prefix), "
            "e.g. attention, patch-embeddings, vit. If omitted, runs all tests."
        ),
    )
    _args = parser.parse_args()
    standalone(_args.test)
