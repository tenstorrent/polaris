#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ---- UNet building blocks ----

# class DoubleConv(nn.Module):
#     """(Conv -> BN -> ReLU) * 2"""
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.net(x)

# class Down(nn.Module):
#     """Downscaling with maxpool then DoubleConv"""
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_ch, out_ch),
#         )

#     def forward(self, x):
#         return self.net(x)

# class Up(nn.Module):
#     """Upscaling then DoubleConv"""
#     def __init__(self, in_ch, out_ch, bilinear=True):
#         super().__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#             self.conv = DoubleConv(in_ch, out_ch)
#         else:
#             # FIX: input channels = in_ch, output channels = in_ch // 2
#             self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_ch, out_ch)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         x1 = F.pad(
#             x1,
#             [diffX // 2, diffX - diffX // 2,
#              diffY // 2, diffY - diffY // 2],
#         )
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

# class OutConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)

# class UNet(nn.Module):
#     def __init__(self, n_channels=3, n_classes=2, bilinear=False):
#         super().__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc   = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1   = Up(1024, 512 // factor, bilinear)
#         self.up2   = Up(512, 256 // factor, bilinear)
#         self.up3   = Up(256, 128 // factor, bilinear)
#         self.up4   = Up(128, 64, bilinear)
#         self.outc  = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x  = self.up1(x5, x4)
#         x  = self.up2(x,  x3)
#         x  = self.up3(x,  x2)
#         x  = self.up4(x,  x1)
#         logits = self.outc(x)
#         return logits

# def export_unet_onnx(
#     batch_size: int = 1,
#     height: int = 256,
#     width: int = 256,
#     n_channels: int = 3,
#     n_classes: int = 2,
#     bilinear: bool = False,
#     output_path: str = "workloads/onnx/unet-fixed-256.onnx",
# ) -> None:
#     model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
#     model.eval()

#     dummy = torch.randn(batch_size, n_channels, height, width)

#     torch.onnx.export(
#         model,
#         dummy,
#         output_path,
#         input_names=["input"],
#         output_names=["logits"],
#         opset_version=18,
#         dynamic_axes=None,
#         do_constant_folding=True,
#     )

#     print(f"Saved UNet ONNX to {output_path}")

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--batch", type=int, default=1)
#     p.add_argument("--height", type=int, default=256)
#     p.add_argument("--width", type=int, default=256)
#     p.add_argument("--n-channels", type=int, default=3)
#     p.add_argument("--n-classes", type=int, default=2)
#     p.add_argument("--bilinear", action="store_true", default=False)
#     p.add_argument(
#         "--output",
#         type=str,
#         default="workloads/onnx/unet-fixed-256.onnx",
#     )
#     args = p.parse_args()

#     export_unet_onnx(
#         batch_size=args.batch,
#         height=args.height,
#         width=args.width,
#         n_channels=args.n_channels,
#         n_classes=args.n_classes,
#         bilinear=args.bilinear,
#         output_path=args.output,
#     )