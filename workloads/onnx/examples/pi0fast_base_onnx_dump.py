# #!/usr/bin/env python
# # SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# # SPDX-License-Identifier: Apache-2.0

# """
# Export the full PI0-FAST (pi0fast-base) model to a fixed-shape ONNX graph
# for Polaris perf projection.

# """

# from pathlib import Path

# import torch
# from torch import Tensor, nn

# try:
#     from lerobot.configs.policies import PreTrainedConfig
#     from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
#     from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPytorch
# except ModuleNotFoundError as e:
#     raise SystemExit(
#         "LeRobot with pi0-fast is not installed.\n"
#         'Install it with:\n'
#         '  pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"\n'
#     ) from e

# MODEL_ID = "lerobot/pi0fast-base"
# DEFAULT_OUT_PATH = "workloads/onnx/pi0fast_base_fixed-224.onnx"

# class Pi0FastForwardWrapper(nn.Module):
#     """
#     Wrapper around the full PI0FastPytorch model so ONNX sees only tensor
#     inputs. Calls the real training forward(), so the entire model graph
#     (PaliGemma + FAST head) is exported.

#     Inputs:
#       image:                   [B, 3, H, W]        float32 in [0, 1]
#       image_mask:              [B]                 bool
#       language_tokens:         [B, L]              int64
#       language_attention_mask: [B, L]              bool
#       fast_action_tokens:      [B, T]              int64
#       fast_action_masks:       [B, T]              bool

#     Output:
#       loss:                    [B]                 float32
#     """

#     def __init__(self, core_model: nn.Module):
#         super().__init__()
#         self.core = core_model

#     def forward(
#         self,
#         image: Tensor,
#         image_mask: Tensor,
#         language_tokens: Tensor,
#         language_attention_mask: Tensor,
#         fast_action_tokens: Tensor,
#         fast_action_masks: Tensor,
#     ) -> Tensor:
#         images = [image]
#         img_masks = [image_mask]

#         loss_dict = self.core(
#             images=images,
#             img_masks=img_masks,
#             tokens=language_tokens,
#             masks=language_attention_mask,
#             fast_action_tokens=fast_action_tokens,
#             fast_action_masks=fast_action_masks,
#         )
#         return loss_dict["loss"]

# def export_pi0fast_onnx(
#     output_path: str = DEFAULT_OUT_PATH,
#     batch_size: int = 1,
# ) -> None:
#     print(f"[pi0fast] Loading PI0FastConfig from {MODEL_ID} ...")
#     cfg_any = PreTrainedConfig.from_pretrained(MODEL_ID)
#     if not isinstance(cfg_any, PI0FastConfig):
#         raise TypeError(f"Expected PI0FastConfig, got {type(cfg_any)}")
#     cfg: PI0FastConfig = cfg_any

#     # Make export deterministic / CPU-friendly
#     cfg.device = "cpu"
#     cfg.gradient_checkpointing = False
#     cfg.compile_model = False
#     cfg.use_kv_cache = False

#     print("[pi0fast] Constructing full PI0FastPytorch model (random weights) ...")
#     core = PI0FastPytorch(
#         config=cfg,
#         rtc_processor=None,
#         paligemma_tokenizer=None,
#     )
#     core.eval()

#     wrapper = Pi0FastForwardWrapper(core)
#     wrapper.eval()

#     img_h, img_w = cfg.image_resolution
#     lang_len = cfg.tokenizer_max_length
#     max_fast_tokens = cfg.max_action_tokens

#     print(
#         "[pi0fast] Using fixed shapes:\n"
#         f"  image                  = ({batch_size}, 3, {img_h}, {img_w})\n"
#         f"  image_mask             = ({batch_size},)\n"
#         f"  language_tokens        = ({batch_size}, {lang_len})\n"
#         f"  language_attention_mask= ({batch_size}, {lang_len})\n"
#         f"  fast_action_tokens     = ({batch_size}, {max_fast_tokens})\n"
#         f"  fast_action_masks      = ({batch_size}, {max_fast_tokens})"
#     )

#     device = torch.device("cpu")

#     dummy_image = torch.zeros(
#         batch_size,
#         3,
#         img_h,
#         img_w,
#         dtype=torch.float32,
#         device=device,
#     )

#     dummy_image_mask = torch.ones(
#         batch_size,
#         dtype=torch.bool,
#         device=device,
#     )

#     dummy_lang_tokens = torch.zeros(
#         batch_size,
#         lang_len,
#         dtype=torch.long,
#         device=device,
#     )

#     dummy_lang_attn = torch.ones(
#         batch_size,
#         lang_len,
#         dtype=torch.bool,
#         device=device,
#     )

#     dummy_fast_tokens = torch.zeros(
#         batch_size,
#         max_fast_tokens,
#         dtype=torch.long,
#         device=device,
#     )

#     dummy_fast_masks = torch.ones(
#         batch_size,
#         max_fast_tokens,
#         dtype=torch.bool,
#         device=device,
#     )

#     out_path = Path(output_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     print(f"[pi0fast] Exporting ONNX to {out_path} ...")
#     torch.onnx.export(
#         wrapper,
#         (
#             dummy_image,
#             dummy_image_mask,
#             dummy_lang_tokens,
#             dummy_lang_attn,
#             dummy_fast_tokens,
#             dummy_fast_masks,
#         ),
#         out_path.as_posix(),
#         input_names=[
#             "image",
#             "image_mask",
#             "language_tokens",
#             "language_attention_mask",
#             "fast_action_tokens",
#             "fast_action_masks",
#         ],
#         output_names=["loss"],
#         opset_version=18,
#         dynamic_axes=None,
#         do_constant_folding=True,
#     )
#     print(f"[pi0fast] Saved fixed-shape ONNX: {out_path}")

# def main() -> None:
#     export_pi0fast_onnx()

# if __name__ == "__main__":
#     main()