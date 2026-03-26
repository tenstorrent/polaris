# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import sys
if __name__ == "__main__":
    sys.path.append(".")

import os
IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
import pytest   # noqa E402
import transformers   # noqa E402
from loguru import logger
if not IS_POLARIS:
    import torch   # noqa E402
    from datasets import load_dataset   # noqa E402
    from transformers import AutoImageProcessor   # noqa E402
    from ttnn.model_preprocessing import preprocess_model_parameters   # noqa E402

    import ttnn   # noqa E402
else:
    import ttsim.front.ttnn as ttnn
    torch = ttnn
    from ttsim.front.ttnn.device import set_default_device
    from ttsim.front.ttnn.tensor import ttnn_random
    import workloads.ttnn.vit.ttnn_functional_vit as ttnn_functional_vit
    import types
    torch_random = ttnn_random
    torch_bfloat16 = ttnn.DataType.BFLOAT16

if not IS_POLARIS:
    from models.common.utility_functions import is_blackhole, is_wormhole_b0, torch_random
    from models.demos.vit.common import load_torch_model
    from models.demos.vit.tt import ttnn_functional_vit
    from tests.ttnn.utils_for_testing import assert_with_pcc

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
            "weight": ttnn.Tensor(shape=weight_shape, dtype=ttnn.DataType.BFLOAT16),
            "bias": ttnn.Tensor(shape=bias_shape, dtype=ttnn.DataType.BFLOAT16)
        })

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

# @pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
# @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])
def test_vit_attention(device, model_name, batch_size, sequence_size):
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
        torch_attention_mask = torch.ones(1, sequence_size, dtype=torch.bfloat16)
        torch_output, *_ = model(torch_hidden_states, torch_attention_mask)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_functional_vit.custom_preprocessor,
        )

        hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
        attention_mask = ttnn.from_torch(torch_attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        config = config_obj
        hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
        attention_mask = ttnn.ones(1, sequence_size, dtype=torch.bfloat16)
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

# @pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
# @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])
@pytest.mark.parametrize("torch_dtype", [torch.bfloat16])
def test_vit_intermediate(device, model_name, batch_size, sequence_size, torch_dtype):
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
            custom_preprocessor=ttnn_functional_vit.custom_preprocessor,
        )

        hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT, device=device)

    else:
        config = config_obj
        hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
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


# @pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
# @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [224])
def test_vit_output(device, model_name, batch_size, sequence_size):
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
            custom_preprocessor=ttnn_functional_vit.custom_preprocessor,
        )

        intermediate = ttnn.from_torch(torch_intermediate, layout=ttnn.TILE_LAYOUT, device=device)
        residual = ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device)

    else:
        config = config_obj
        intermediate = torch_random(
            (batch_size, sequence_size, config.intermediate_size), -0.1, 0.1, dtype=torch.bfloat16
        )
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

def standalone():
    from ttsim.front.ttnn.device import open_device
    device = open_device()
    logger.warning('1')
    run_vit_attention("vit-attention", device, {})

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    standalone()
