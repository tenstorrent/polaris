#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
NMS-Free BBox Coder for BEVFormer - TTSim Implementation

This module provides bounding box decoding without Non-Maximum Suppression (NMS)
for BEVFormer 3D object detection.

Original PyTorch implementation: projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py
Converted to TTSim: February 9, 2026

The NMS-Free decoder processes classification scores and bounding box predictions
from the detection head, selecting the top-K detections and filtering them based
on score thresholds and center range constraints.
"""

import sys
import os
from loguru import logger

# Add paths for TTSim
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
from ttsim.front.functional.sim_nn import Module


class NMSFreeCoder(Module):
    """
    Bbox coder for NMS-free detector.

    This coder decodes bounding box predictions without requiring Non-Maximum
    Suppression (NMS). It uses top-K selection and threshold-based filtering
    to produce final detections.

    Args:
        name (str): Module name
        pc_range (list[float]): Range of point cloud [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size (list[float], optional): Size of voxels. Default: None.
        post_center_range (list[float], optional): Limit of the center coordinates.
            Format: [x_min, y_min, z_min, x_max, y_max, z_max]. Default: None.
        max_num (int): Max number of detections to keep. Default: 100.
        score_threshold (float, optional): Threshold to filter boxes based on score.
            Default: None.
        num_classes (int): Number of object classes. Default: 10.

    Original PyTorch code:
        from mmdet.core.bbox import BaseBBoxCoder
        from mmdet.core.bbox.builder import BBOX_CODERS

    TTSim equivalent:
        - BaseBBoxCoder → ttsim.front.functional.sim_nn.Module
        - BBOX_CODERS.register_module() → Not needed (no registry in TTSim)
    """

    def __init__(
        self,
        name,
        pc_range,
        voxel_size=None,
        post_center_range=None,
        max_num=100,
        score_threshold=None,
        num_classes=10,
    ):
        super().__init__()
        self.name = name
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        """
        Encode method (not used in inference).

        In the original implementation, this is a placeholder method
        as encoding is not required for the NMS-free decoder.
        """
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """
        Decode bounding boxes for a single sample.

        This method processes classification scores and bbox predictions to:
        1. Apply sigmoid activation to scores
        2. Select top-K detections based on scores
        3. Denormalize bbox coordinates
        4. Filter detections by score threshold
        5. Filter detections by center range

        Args:
            cls_scores: Classification scores
                Shape: [num_query, cls_out_channels]
                Note: cls_out_channels includes all classes (no background class)
            bbox_preds: Normalized bounding box predictions
                Shape: [num_query, code_size]
                Format: (cx, cy, w_log, l_log, cz, h_log, rot_sine, rot_cosine, vx, vy)
                Where dimensions are log-transformed and rotation is sin/cos decomposed

        Returns:
            dict: Decoded boxes with keys:
                - 'bboxes': Final bounding boxes [N, code_size]
                            Format: (cx, cy, cz, w, l, h, rot, vx, vy)
                - 'scores': Detection scores [N]
                - 'labels': Class labels [N]

        PyTorch operations → TTSim equivalents:
            - cls_scores.sigmoid() → F.Sigmoid()
            - .view(-1) → F.Reshape()
            - .topk(max_num) → F.TopK()
            - indexs % self.num_classes → F.Mod()
            - indexs // self.num_classes → F.Div() + F.Floor()
            - bbox_preds[bbox_index] → F.Gather()
            - denormalize_bbox() → Custom TTSim function
            - comparisons (>, >=, <=) → F.Greater(), F.GreaterOrEqual(), F.LessOrEqual()
            - logical operations (&) → F.And()
            - boolean masking → F.Where() or custom gather
        """
        import time

        unique_id = str(int(time.time() * 1000000) % 1000000)  # Microsecond timestamp

        max_num = self.max_num

        # Apply sigmoid to classification scores
        # PyTorch: cls_scores = cls_scores.sigmoid()
        cls_scores_sigmoid = F.Sigmoid(f"{self.name}.sigmoid_{unique_id}")(cls_scores)

        # Flatten scores and get top-K
        # PyTorch: scores, indexs = cls_scores.view(-1).topk(max_num)
        # Shape: [num_query, cls_out_channels] → [num_query * cls_out_channels]
        # Get shape values from cls_scores tensor
        num_queries = int(cls_scores.shape[0])
        cls_out_channels = int(cls_scores.shape[1])

        # Create reshape operation with explicit shape calculation
        reshape_op = F.Reshape(f"{self.name}.decode.reshape_flat_{unique_id}")
        shape_tensor = F._from_data(
            f"{self.name}.decode.shape_flat_{unique_id}",
            np.array([num_queries * cls_out_channels], dtype=np.int64),
            is_const=True,
        )

        scores_flat = reshape_op(cls_scores_sigmoid, shape_tensor)

        # TopK to get top max_num scores and indices
        # TopK returns (values, indices)
        scores_top, indices_top = F.topk(
            f"{self.name}.decode.topk_{unique_id}",
            k=max_num,
            axis=0,
            largest=True,
            sorted=True,
        )(scores_flat)

        # Compute labels and bbox indices
        # PyTorch: labels = indexs % self.num_classes
        # PyTorch: bbox_index = indexs // self.num_classes
        num_classes_tensor = F._from_data(
            f"{self.name}.num_classes",
            np.array([self.num_classes], dtype=np.int64),
            is_const=True,
        )

        # Convert int64 indices to float for arithmetic, then back
        indices_float = F.Cast(f"{self.name}.indices_to_float", to="Float")(indices_top)
        num_classes_float = F.Cast(f"{self.name}.num_classes_to_float", to="Float")(
            num_classes_tensor
        )

        # Compute labels using modulo
        labels_float = F.Mod(f"{self.name}.mod")(indices_float, num_classes_float)
        labels = F.Cast(f"{self.name}.labels_to_int", to="Int64")(labels_float)

        # Compute bbox_index using floor division
        bbox_index_float = F.Div(f"{self.name}.div")(indices_float, num_classes_float)
        bbox_index_floor = F.Floor(f"{self.name}.floor")(bbox_index_float)
        bbox_index = F.Cast(f"{self.name}.bbox_index_to_int", to="Int64")(
            bbox_index_floor
        )

        # Gather bbox predictions using bbox_index
        # PyTorch: bbox_preds = bbox_preds[bbox_index]
        # bbox_preds shape: [num_query, code_size]
        # bbox_index shape: [max_num]
        # Output shape: [max_num, code_size]
        bbox_preds_selected = F.Gather(f"{self.name}.gather_bboxes", axis=0)(
            bbox_preds, bbox_index
        )

        # Denormalize bounding boxes
        # PyTorch: final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_box_preds = self.denormalize_bbox(bbox_preds_selected, self.pc_range)

        # Assign final scores and labels
        final_scores = scores_top
        final_preds = labels

        # Apply score threshold filtering
        # PyTorch: if self.score_threshold is not None:
        #              thresh_mask = final_scores > self.score_threshold
        if self.score_threshold is not None:
            # Note: In PyTorch, there's dynamic threshold adjustment logic
            # For TTSim inference, we'll use a fixed threshold
            # The dynamic adjustment (tmp_score *= 0.9) is for training/validation
            # and requires iterative logic which is not suitable for static graphs

            threshold_tensor = F._from_data(
                f"{self.name}.threshold",
                np.array([self.score_threshold], dtype=np.float32),
                is_const=True,
            )

            # thresh_mask = final_scores > threshold
            thresh_mask = F.Greater(f"{self.name}.threshold_mask")(
                final_scores, threshold_tensor
            )
        else:
            # If no threshold, all detections pass
            thresh_mask = F._from_data(
                f"{self.name}.all_pass", np.ones([max_num], dtype=bool), is_const=True
            )

        # Apply post center range filtering
        # PyTorch: if self.post_center_range is not None:
        #              mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)
        #              mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1)
        if self.post_center_range is not None:
            pc_range_min = F._from_data(
                f"{self.name}.pc_range_min",
                np.array(self.post_center_range[:3], dtype=np.float32).reshape(1, 3),
                is_const=True,
            )
            pc_range_max = F._from_data(
                f"{self.name}.pc_range_max",
                np.array(self.post_center_range[3:], dtype=np.float32).reshape(1, 3),
                is_const=True,
            )

            # Extract center coordinates (first 3 elements: cx, cy, cz)
            # final_box_preds shape: [max_num, code_size]
            # We need [..., :3] which is [max_num, 3]
            starts_0 = F._from_data(
                f"{self.name}.starts_0", np.array([0], dtype=np.int64), is_const=True
            )
            ends_3 = F._from_data(
                f"{self.name}.ends_3", np.array([3], dtype=np.int64), is_const=True
            )
            axes_1 = F._from_data(
                f"{self.name}.axes_1", np.array([1], dtype=np.int64), is_const=True
            )
            steps_1 = F._from_data(
                f"{self.name}.steps_1", np.array([1], dtype=np.int64), is_const=True
            )

            centers = F.SliceF(f"{self.name}.slice_centers", out_shape=[max_num, 3])(
                final_box_preds, starts_0, ends_3, axes_1, steps_1
            )

            # Check if centers >= pc_range_min
            mask_min = F.GreaterOrEqual(f"{self.name}.mask_min")(centers, pc_range_min)
            # Check if centers <= pc_range_max
            mask_max = F.LessOrEqual(f"{self.name}.mask_max")(centers, pc_range_max)

            # Combine masks: all must be true along axis 1
            # mask_min and mask_max shape: [max_num, 3]
            # We need to reduce to [max_num] by checking if all are true
            mask_combined = F.And(f"{self.name}.mask_combined")(mask_min, mask_max)

            # Reduce along axis 1 to get [max_num] boolean mask
            # all(1) in PyTorch means all elements along dim=1 must be True
            mask_range = F.ReduceMin(f"{self.name}.mask_range", axis=1, keepdims=False)(
                F.Cast(f"{self.name}.mask_to_int", to="Int32")(mask_combined)
            )
            mask_range_bool = F.Cast(f"{self.name}.mask_to_bool", to="Bool")(mask_range)

            # Combine with threshold mask
            # PyTorch: if self.score_threshold: mask &= thresh_mask
            if self.score_threshold is not None:
                final_mask = F.And(f"{self.name}.final_mask")(
                    mask_range_bool, thresh_mask
                )
            else:
                final_mask = mask_range_bool

            # Apply mask to get final detections
            # PyTorch:
            #   boxes3d = final_box_preds[mask]
            #   scores = final_scores[mask]
            #   labels = final_preds[mask]

            # In TTSim, we can use NonZero + Gather, or Compress
            # Let's use Compress which is like boolean indexing
            # Note: Compress might not be available, so we'll use Where for conditional selection
            # or use a gather-based approach

            # For simplicity in TTSim inference, we can use a gather-based filtering
            # 1. Find indices where mask is True using NonZero
            # 2. Gather elements at those indices

            # NonZero returns indices where condition is true
            mask_indices = F.NonZero(f"{self.name}.nonzero")(final_mask)

            # Gather filtered results
            # Note: NonZero output is 2D [rank, num_true], we need to transpose and squeeze
            # For a 1D input, output is [1, num_true], so we squeeze axis 0
            axes_squeeze = F._from_data(
                f"{self.name}.axes_squeeze",
                np.array([0], dtype=np.int64),
                is_const=True,
            )
            mask_indices_1d = F.Squeeze(f"{self.name}.squeeze_indices")(
                mask_indices, axes_squeeze
            )

            boxes3d = F.Gather(f"{self.name}.gather_boxes", axis=0)(
                final_box_preds, mask_indices_1d
            )
            scores = F.Gather(f"{self.name}.gather_scores", axis=0)(
                final_scores, mask_indices_1d
            )
            labels_out = F.Gather(f"{self.name}.gather_labels", axis=0)(
                final_preds, mask_indices_1d
            )

            predictions_dict = {
                "bboxes": boxes3d,
                "scores": scores,
                "labels": labels_out,
            }
        else:
            # If no post_center_range, this is not implemented in original code
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )

        return predictions_dict

    def denormalize_bbox(self, normalized_bboxes, pc_range):
        """
        Denormalize bounding boxes.

        Converts normalized bbox predictions back to absolute coordinates.
        Reverses the transformations: exp(log(dim)) → dim and (sin, cos) → angle

        Args:
            normalized_bboxes: Normalized bboxes [N, code_size]
                Format: (cx, cy, w_log, l_log, cz, h_log, rot_sine, rot_cosine, vx, vy)
            pc_range: Point cloud range (not used in this version but kept for compatibility)

        Returns:
            Denormalized bboxes [N, code_size]
            Format: (cx, cy, cz, w, l, h, rot, vx, vy)

        PyTorch operations:
            - normalized_bboxes[..., 6:7] → F.SliceF()
            - torch.atan2(rot_sine, rot_cosine) → F.Atan2()
            - w.exp() → F.Exp()
            - torch.cat([...], dim=-1) → F.ConcatX()
        """
        name = f"{self.name}.denorm"

        # Extract components using slice
        # Rotation components: indices 6 (sine) and 7 (cosine)
        starts_6 = F._from_data(
            f"{name}.starts_6", np.array([6], dtype=np.int64), is_const=True
        )
        starts_7 = F._from_data(
            f"{name}.starts_7", np.array([7], dtype=np.int64), is_const=True
        )
        ends_7 = F._from_data(
            f"{name}.ends_7", np.array([7], dtype=np.int64), is_const=True
        )
        ends_8 = F._from_data(
            f"{name}.ends_8", np.array([8], dtype=np.int64), is_const=True
        )
        axes_1 = F._from_data(
            f"{name}.axes_1", np.array([1], dtype=np.int64), is_const=True
        )
        steps_1 = F._from_data(
            f"{name}.steps_1", np.array([1], dtype=np.int64), is_const=True
        )

        N = normalized_bboxes.shape[0]
        code_size = normalized_bboxes.shape[1]

        # Extract rotation sine and cosine
        rot_sine = F.SliceF(f"{name}.rot_sine", out_shape=[N, 1])(
            normalized_bboxes, starts_6, ends_7, axes_1, steps_1
        )
        rot_cosine = F.SliceF(f"{name}.rot_cosine", out_shape=[N, 1])(
            normalized_bboxes, starts_7, ends_8, axes_1, steps_1
        )

        # Compute rotation angle using atan2
        rot = F.Atan2(f"{name}.atan2")(rot_sine, rot_cosine)

        # Extract center coordinates
        starts_0 = F._from_data(
            f"{name}.starts_0", np.array([0], dtype=np.int64), is_const=True
        )
        starts_1 = F._from_data(
            f"{name}.starts_1", np.array([1], dtype=np.int64), is_const=True
        )
        starts_2 = F._from_data(
            f"{name}.starts_2", np.array([2], dtype=np.int64), is_const=True
        )
        starts_3 = F._from_data(
            f"{name}.starts_3", np.array([3], dtype=np.int64), is_const=True
        )
        starts_4 = F._from_data(
            f"{name}.starts_4", np.array([4], dtype=np.int64), is_const=True
        )
        starts_5 = F._from_data(
            f"{name}.starts_5", np.array([5], dtype=np.int64), is_const=True
        )

        ends_1 = F._from_data(
            f"{name}.ends_1", np.array([1], dtype=np.int64), is_const=True
        )
        ends_2 = F._from_data(
            f"{name}.ends_2", np.array([2], dtype=np.int64), is_const=True
        )
        ends_3 = F._from_data(
            f"{name}.ends_3", np.array([3], dtype=np.int64), is_const=True
        )
        ends_4 = F._from_data(
            f"{name}.ends_4", np.array([4], dtype=np.int64), is_const=True
        )
        ends_5 = F._from_data(
            f"{name}.ends_5", np.array([5], dtype=np.int64), is_const=True
        )
        ends_6 = F._from_data(
            f"{name}.ends_6", np.array([6], dtype=np.int64), is_const=True
        )

        cx = F.SliceF(f"{name}.cx", out_shape=[N, 1])(
            normalized_bboxes, starts_0, ends_1, axes_1, steps_1
        )
        cy = F.SliceF(f"{name}.cy", out_shape=[N, 1])(
            normalized_bboxes, starts_1, ends_2, axes_1, steps_1
        )
        cz = F.SliceF(f"{name}.cz", out_shape=[N, 1])(
            normalized_bboxes, starts_4, ends_5, axes_1, steps_1
        )

        # Extract size (log-transformed)
        w_log = F.SliceF(f"{name}.w_log", out_shape=[N, 1])(
            normalized_bboxes, starts_2, ends_3, axes_1, steps_1
        )
        l_log = F.SliceF(f"{name}.l_log", out_shape=[N, 1])(
            normalized_bboxes, starts_3, ends_4, axes_1, steps_1
        )
        h_log = F.SliceF(f"{name}.h_log", out_shape=[N, 1])(
            normalized_bboxes, starts_5, ends_6, axes_1, steps_1
        )

        # Apply exp to reverse log transform
        w = F.Exp(f"{name}.w_exp")(w_log)
        l = F.Exp(f"{name}.l_exp")(l_log)
        h = F.Exp(f"{name}.h_exp")(h_log)

        # Check if velocity is present (code_size > 8)
        if code_size > 8:
            # Extract velocity
            starts_8 = F._from_data(
                f"{name}.starts_8", np.array([8], dtype=np.int64), is_const=True
            )
            starts_9 = F._from_data(
                f"{name}.starts_9", np.array([9], dtype=np.int64), is_const=True
            )
            ends_9 = F._from_data(
                f"{name}.ends_9", np.array([9], dtype=np.int64), is_const=True
            )
            ends_10 = F._from_data(
                f"{name}.ends_10", np.array([10], dtype=np.int64), is_const=True
            )

            vx = F.SliceF(f"{name}.vx", out_shape=[N, 1])(
                normalized_bboxes, starts_8, ends_9, axes_1, steps_1
            )
            vy = F.SliceF(f"{name}.vy", out_shape=[N, 1])(
                normalized_bboxes, starts_9, ends_10, axes_1, steps_1
            )

            # Concatenate: [cx, cy, cz, w, l, h, rot, vx, vy]
            denormalized_bboxes = F.ConcatX(f"{name}.concat", axis=-1)(
                cx, cy, cz, w, l, h, rot, vx, vy
            )
        else:
            # No velocity: [cx, cy, cz, w, l, h, rot]
            denormalized_bboxes = F.ConcatX(f"{name}.concat", axis=-1)(
                cx, cy, cz, w, l, h, rot
            )

        return denormalized_bboxes

    def decode(self, preds_dicts):
        """
        Decode bounding boxes for a batch of samples.

        Args:
            preds_dicts (dict): Predictions dictionary with keys:
                - 'all_cls_scores': Classification scores
                    Shape: [num_decoder_layers, bs, num_query, cls_out_channels]
                - 'all_bbox_preds': Bounding box predictions
                    Shape: [num_decoder_layers, bs, num_query, code_size]

        Returns:
            list[dict]: List of decoded predictions for each sample in batch.
                Each dict contains 'bboxes', 'scores', 'labels' keys.

        Note: This function processes the last decoder layer outputs (index -1)
        and iterates over the batch dimension.
        """
        # Get predictions from last decoder layer
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]

        batch_size = all_cls_scores.shape[0]
        predictions_list = []

        for i in range(batch_size):
            # Slice batch dimension to get single sample
            # Note: In TTSim inference, batch processing might be handled differently
            # For a static graph, we would process the full batch, but the original
            # PyTorch code processes samples individually

            # Extract i-th sample
            starts_i = F._from_data(
                f"{self.name}.decode.starts_{i}",
                np.array([i], dtype=np.int64),
                is_const=True,
            )
            ends_i_plus_1 = F._from_data(
                f"{self.name}.decode.ends_{i}",
                np.array([i + 1], dtype=np.int64),
                is_const=True,
            )
            axes_0 = F._from_data(
                f"{self.name}.decode.axes_0",
                np.array([0], dtype=np.int64),
                is_const=True,
            )
            steps_1 = F._from_data(
                f"{self.name}.decode.steps_1",
                np.array([1], dtype=np.int64),
                is_const=True,
            )

            num_query = all_cls_scores.shape[1]
            cls_out_channels = all_cls_scores.shape[2]
            code_size = all_bbox_preds.shape[2]

            cls_scores_i = F.SliceF(
                f"{self.name}.decode.cls_scores_{i}",
                out_shape=[1, num_query, cls_out_channels],
            )(all_cls_scores, starts_i, ends_i_plus_1, axes_0, steps_1)

            bbox_preds_i = F.SliceF(
                f"{self.name}.decode.bbox_preds_{i}",
                out_shape=[1, num_query, code_size],
            )(all_bbox_preds, starts_i, ends_i_plus_1, axes_0, steps_1)

            # Squeeze batch dimension - axes must be passed as input tensor
            axes_squeeze_0 = F._from_data(
                f"{self.name}.decode.axes_squeeze_{i}",
                np.array([0], dtype=np.int64),
                is_const=True,
            )
            cls_scores_i_squeezed = F.Squeeze(f"{self.name}.decode.squeeze_cls_{i}")(
                cls_scores_i, axes_squeeze_0
            )
            bbox_preds_i_squeezed = F.Squeeze(f"{self.name}.decode.squeeze_bbox_{i}")(
                bbox_preds_i, axes_squeeze_0
            )

            # Decode single sample
            predictions = self.decode_single(
                cls_scores_i_squeezed, bbox_preds_i_squeezed
            )
            predictions_list.append(predictions)

        return predictions_list


# ============================================================================
# Standalone Functions (for use outside Module class)
# ============================================================================


def denormalize_bbox_standalone(normalized_bboxes, pc_range=None):
    """
    Standalone function for denormalizing bounding boxes.

    This provides a functional interface for bbox denormalization
    without requiring a NMSFreeCoder module instance.

    Args:
        normalized_bboxes: TTSim tensor [N, code_size]
            Format: (cx, cy, w_log, l_log, cz, h_log, rot_sine, rot_cosine, vx, vy)
        pc_range: Point cloud range (unused, kept for API compatibility)

    Returns:
        Denormalized bboxes TTSim tensor [N, code_size]
        Format: (cx, cy, cz, w, l, h, rot, vx, vy)
    """
    # Create a temporary coder instance for denormalization
    temp_coder = NMSFreeCoder(
        name="denorm_standalone",
        pc_range=pc_range if pc_range is not None else [0, 0, 0, 0, 0, 0],
        max_num=100,
        num_classes=10,
    )

    return temp_coder.denormalize_bbox(normalized_bboxes, pc_range)


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("NMS-Free BBox Coder - TTSim Implementation")
    logger.info("=" * 80)
    logger.info("\n✓ Module imported successfully!")
    logger.info("\nKey components:")
    logger.info("  - NMSFreeCoder: Main decoder class")
    logger.info("  - decode_single: Decode single sample")
    logger.info("  - decode: Decode batch of samples")
    logger.info("  - denormalize_bbox: Denormalize bbox predictions")
    logger.info("\nConversion from PyTorch to TTSim:")
    logger.info("  ✓ BaseBBoxCoder → Module")
    logger.info("  ✓ sigmoid() → F.Sigmoid()")
    logger.info("  ✓ view(-1) → F.Reshape()")
    logger.info("  ✓ topk() → F.TopK()")
    logger.info("  ✓ % operator → F.Mod()")
    logger.info("  ✓ // operator → F.Div() + F.Floor()")
    logger.info("  ✓ tensor indexing → F.Gather()")
    logger.info("  ✓ atan2() → F.Atan2()")
    logger.info("  ✓ exp() → F.Exp()")
    logger.info("  ✓ Boolean masking → F.NonZero() + F.Gather()")
    logger.info("\n✓ All operations converted to TTSim!")
    logger.info("=" * 80)
