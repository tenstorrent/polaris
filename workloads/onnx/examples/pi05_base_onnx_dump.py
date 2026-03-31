# #!/usr/bin/env python
# # SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# # SPDX-License-Identifier: Apache-2.0

# """
# Export the full Pi0.5 (pi05_base) model to a fixed-shape ONNX graph
# for Polaris perf projection.

# """

# from pathlib import Path

# import torch
# from torch import Tensor, nn

# try:
#     from lerobot.configs.policies import PreTrainedConfig
#     from lerobot.policies.pi05.configuration_pi05 import PI05Config
#     from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
# except ModuleNotFoundError as e: 
#     raise SystemExit(
#         "LeRobot with pi05 is not installed.\n"
#         'Install it with:\n'
#         '  pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"\n'
#     ) from e

# MODEL_ID = "lerobot/pi05_base"
# DEFAULT_OUT_PATH = "workloads/onnx/pi05_base_fixed-224.onnx"

# class PI05PytorchExport(PI05Pytorch):

#     def sample_noise(self, shape, device):
#         return torch.zeros(shape, dtype=torch.float32, device=device)

#     def sample_time(self, bsize: int, device):
#         return torch.full(
#             (bsize,),
#             0.5,
#             dtype=torch.float32,
#             device=device,
#         )

# class Pi05ForwardWrapper(nn.Module):
#     """
#     Wrapper around the full PI05Pytorch model so ONNX sees only tensor
#     inputs. Calls the real training forward(), so the entire model graph
#     (PaliGemma + flow-matching head) is exported.

#     Inputs:
#       image:                   [B, 3, H, W]        float32 in [0, 1]
#       image_mask:              [B]                 bool
#       language_tokens:         [B, L]              int64
#       language_attention_mask: [B, L]              bool
#       actions:                 [B, T, D]           float32

#     Output:
#       loss:                    [B, T, D]           float32 (per-step, per-dim loss)
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
#         actions: Tensor,
#     ) -> Tensor:
#         images = [image]
#         img_masks = [image_mask]

#         losses = self.core(
#             images=images,
#             img_masks=img_masks,
#             tokens=language_tokens,
#             masks=language_attention_mask,
#             actions=actions,
#         )
#         return losses

# def export_pi05_onnx(
#     output_path: str = DEFAULT_OUT_PATH,
#     batch_size: int = 1,
# ) -> None:
#     print(f"[pi05] Loading PI05Config from {MODEL_ID} ...")
#     cfg_any = PreTrainedConfig.from_pretrained(MODEL_ID)
#     if not isinstance(cfg_any, PI05Config):
#         raise TypeError(f"Expected PI05Config, got {type(cfg_any)}")
#     cfg: PI05Config = cfg_any

#     cfg.device = "cpu"
#     cfg.gradient_checkpointing = False
#     cfg.compile_model = False

#     print("[pi05] Constructing full PI05PytorchExport model (random weights) ...")
#     core = PI05PytorchExport(
#         config=cfg,
#         rtc_processor=None,
#     )
#     core.eval()

#     wrapper = Pi05ForwardWrapper(core)
#     wrapper.eval()

#     img_h, img_w = cfg.image_resolution
#     lang_len = cfg.tokenizer_max_length
#     chunk_size = cfg.chunk_size
#     max_action_dim = cfg.max_action_dim

#     print(
#         "[pi05] Using fixed shapes:\n"
#         f"  image                  = ({batch_size}, 3, {img_h}, {img_w})\n"
#         f"  image_mask             = ({batch_size},)\n"
#         f"  language_tokens        = ({batch_size}, {lang_len})\n"
#         f"  language_attention_mask= ({batch_size}, {lang_len})\n"
#         f"  actions                = ({batch_size}, {chunk_size}, {max_action_dim})"
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

#     dummy_actions = torch.zeros(
#         batch_size,
#         chunk_size,
#         max_action_dim,
#         dtype=torch.float32,
#         device=device,
#     )

#     out_path = Path(output_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     print(f"[pi05] Exporting ONNX to {out_path} ...")
#     torch.onnx.export(
#         wrapper,
#         (
#             dummy_image,
#             dummy_image_mask,
#             dummy_lang_tokens,
#             dummy_lang_attn,
#             dummy_actions,
#         ),
#         out_path.as_posix(),
#         input_names=[
#             "image",
#             "image_mask",
#             "language_tokens",
#             "language_attention_mask",
#             "actions",
#         ],
#         output_names=["loss"],
#         opset_version=18,
#         dynamic_axes=None,
#         do_constant_folding=True,
#     )
#     print(f"[pi05] Saved fixed-shape ONNX: {out_path}")

# def main() -> None:
#     export_pi05_onnx()

# if __name__ == "__main__":
#     main()