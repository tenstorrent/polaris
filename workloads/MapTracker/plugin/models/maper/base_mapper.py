#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
ttsim version of BaseMapper for MapTracker
Uses ttsim operators for graph simulation
"""

# -------------------------------PyTorch--------------------------------

# from abc import ABCMeta, abstractmethod
#
# import torch.nn as nn
# from mmcv.runner import auto_fp16
# from mmcv.utils import print_log
#
# from mmdet.utils import get_root_logger
# from mmdet3d.models.builder import DETECTORS
#
# MAPPERS = DETECTORS
#
# class BaseMapper(nn.Module, metaclass=ABCMeta):
#     """Base class for mappers."""
#
#     def __init__(self):
#         super(BaseMapper, self).__init__()
#         self.fp16_enabled = False
#
#     @property
#     def with_neck(self):
#         """bool: whether the detector has a neck"""
#         return hasattr(self, 'neck') and self.neck is not None
#
#     # TODO: these properties need to be carefully handled
#     # for both single stage & two stage detectors
#     @property
#     def with_shared_head(self):
#         """bool: whether the detector has a shared head in the RoI Head"""
#         return hasattr(self, 'roi_head') and self.roi_head.with_shared_head
#
#     @property
#     def with_bbox(self):
#         """bool: whether the detector has a bbox head"""
#         return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
#                 or (hasattr(self, 'bbox_head') and self.bbox_head is not None))
#
#     @property
#     def with_mask(self):
#         """bool: whether the detector has a mask head"""
#         return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
#                 or (hasattr(self, 'mask_head') and self.mask_head is not None))
#
#     #@abstractmethod
#     def extract_feat(self, imgs):
#         """Extract features from images."""
#         pass
#
#     def forward_train(self, *args, **kwargs):
#         pass
#
#     #@abstractmethod
#     def simple_test(self, img, img_metas, **kwargs):
#         pass
#
#     #@abstractmethod
#     def aug_test(self, imgs, img_metas, **kwargs):
#         """Test function with test time augmentation."""
#         pass
#
#     def init_weights(self, pretrained=None):
#         """Initialize the weights in detector.
#
#         Args:
#             pretrained (str, optional): Path to pre-trained weights.
#                 Defaults to None.
#         """
#         if pretrained is not None:
#             logger = get_root_logger()
#             print_log(f'load model from: {pretrained}', logger=logger)
#
#     def forward_test(self, *args, **kwargs):
#         """
#         Args:
#         """
#         if True:
#             self.simple_test()
#         else:
#             self.aug_test()
#
#     # @auto_fp16(apply_to=('img', ))
#     def forward(self, *args, return_loss=True, **kwargs):
#         """Calls either :func:`forward_train` or :func:`forward_test` depending
#         on whether ``return_loss`` is ``True``.
#
#         Note this setting will change the expected inputs. When
#         ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
#         and List[dict]), and when ``resturn_loss=False``, img and img_meta
#         should be double nested (i.e.  List[Tensor], List[List[dict]]), with
#         the outer list indicating test time augmentations.
#         """
#         if return_loss:
#             return self.forward_train(*args, **kwargs)
#         else:
#             kwargs.pop('rescale')
#             return self.forward_test(*args, **kwargs)
#
#     def train_step(self, data_dict, optimizer):
#         """The iteration step during training.
#
#         This method defines an iteration step during training, except for the
#         back propagation and optimizer updating, which are done in an optimizer
#         hook. Note that in some complicated cases or models, the whole process
#         including back propagation and optimizer updating is also defined in
#         this method, such as GAN.
#
#         Args:
#             data_dict (dict): The output of dataloader.
#             optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
#                 runner is passed to ``train_step()``. This argument is unused
#                 and reserved.
#
#         Returns:
#             dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
#                 ``num_samples``.
#
#                 - ``loss`` is a tensor for back propagation, which can be a \
#                 weighted sum of multiple losses.
#                 - ``log_vars`` contains all the variables to be sent to the
#                 logger.
#                 - ``num_samples`` indicates the batch size (when the model is \
#                 DDP, it means the batch size on each GPU), which is used for \
#                 averaging the logs.
#         """
#         loss, log_vars, num_samples = self(**data_dict)
#
#         outputs = dict(
#             loss=loss, log_vars=log_vars, num_samples=num_samples)
#
#         return outputs
#
#     def val_step(self, data, optimizer):
#         """The iteration step during validation.
#
#         This method shares the same signature as :func:`train_step`, but used
#         during val epochs. Note that the evaluation after training epochs is
#         not implemented with this method, but an evaluation hook.
#         """
#         loss, log_vars, num_samples = self(**data)
#
#         outputs = dict(
#             loss=loss, log_vars=log_vars, num_samples=num_samples)
#
#         return outputs
#
#     def show_result(self,
#                     **kwargs):
#         img = None
#         return img

# -------------------------------TTSIM-----------------------------------


from abc import ABCMeta, abstractmethod
import ttsim.front.functional.sim_nn as SimNN
import logging


class BaseMapper(SimNN.Module, metaclass=ABCMeta):
    """
    Base class for mappers (ttsim version)

    This is a ttsim implementation using SimNN.Module as base.
    Provides the core mapper interface for training and validation.
    """

    def __init__(self):
        """Initialize the base mapper"""
        super(BaseMapper, self).__init__()
        self.name = "base_mapper"  # Required by SimNN.Module
        self.fp16_enabled = False
        self.logger = logging.getLogger(__name__)

    # ============================================================
    # Property Methods - Check for submodule existence
    # ============================================================

    @property
    def with_neck(self):
        """
        Check if the mapper has a neck module

        Returns:
            bool: True if neck exists and is not None
        """
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_shared_head(self):
        """
        Check if the mapper has a shared head in the RoI Head

        Returns:
            bool: True if roi_head exists and has shared_head
        """
        return hasattr(self, "roi_head") and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """
        Check if the mapper has a bounding box head

        Returns:
            bool: True if bbox head exists (in roi_head or standalone)
        """
        return (hasattr(self, "roi_head") and self.roi_head.with_bbox) or (
            hasattr(self, "bbox_head") and self.bbox_head is not None
        )

    @property
    def with_mask(self):
        """
        Check if the mapper has a mask head

        Returns:
            bool: True if mask head exists (in roi_head or standalone)
        """
        return (hasattr(self, "roi_head") and self.roi_head.with_mask) or (
            hasattr(self, "mask_head") and self.mask_head is not None
        )

    # ============================================================
    # Abstract Methods - To be implemented by subclasses
    # ============================================================

    @abstractmethod
    def extract_feat(self, imgs):
        """
        Extract features from images

        Args:
            imgs (SimTensor): Input images

        Returns:
            SimTensor or tuple: Extracted features
        """
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Forward function for training mode

        Args:
            imgs (SimTensor): Input images
            img_metas (list[dict]): Meta information for each image
            **kwargs: Additional arguments (targets, etc.)

        Returns:
            dict: Dictionary containing:
                - loss (SimTensor): Total loss for backpropagation
                - log_vars (dict): Variables to log (using .data values)
                - num_samples (int): Number of samples in batch
        """
        pass

    @abstractmethod
    def simple_test(self, imgs, img_metas, **kwargs):
        """
        Simple test (single-scale, no augmentation)

        Args:
            imgs (SimTensor): Input images
            img_metas (list[dict]): Meta information for each image
            **kwargs: Additional arguments

        Returns:
            list: Detection/segmentation results for each image
        """
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        """
        Test with augmentation (optional, can be overridden)

        Args:
            imgs (list[SimTensor]): Multi-scale images
            img_metas (list[list[dict]]): Multi-scale meta information
            **kwargs: Additional arguments

        Returns:
            list: Augmented test results
        """
        raise NotImplementedError("Augmentation test not implemented")

    # ============================================================
    # Initialization and Setup
    # ============================================================

    def init_weights(self, pretrained=None):
        """Initialize weights (placeholder for TTSim inference).

        In TTSim, weights are loaded from pre-trained PyTorch checkpoints
        by copying .data attributes onto the corresponding SimTensor
        parameters. This method exists for API compatibility.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Not used in TTSim — weight loading is handled externally.
        """
        # Initialization handled by PyTorch model loading or weight copying
        pass

    # ============================================================
    # Forward Pass Dispatchers
    # ============================================================

    def forward_test(self, imgs, img_metas, use_aug=False, **kwargs):
        """
        Forward function for testing

        Args:
            imgs (SimTensor or list[SimTensor]): Images for testing
            img_metas (list[dict] or list[list[dict]]): Meta information
            use_aug (bool): Whether to use test-time augmentation
            **kwargs: Additional arguments

        Returns:
            list: Test results
        """
        if use_aug:
            return self.aug_test(imgs, img_metas, **kwargs)
        else:
            return self.simple_test(imgs, img_metas, **kwargs)

    def forward(self, imgs, img_metas=None, return_loss=True, **kwargs):
        """
        Main forward function - dispatches to train or test

        Calls either forward_train() or forward_test() depending on return_loss flag.

        Args:
            imgs (SimTensor): Input images
            img_metas (list[dict], optional): Meta information for each image
            return_loss (bool): If True, run training mode. If False, run test mode.
                Default: True
            **kwargs: Additional arguments passed to train/test functions

        Returns:
            dict or list:
                - If return_loss=True: dict with loss (SimTensor), log_vars, num_samples
                - If return_loss=False: list of test results
        """
        if return_loss:
            # Training mode
            return self.forward_train(imgs, img_metas, **kwargs)
        else:
            # Testing mode
            # Remove 'rescale' if present (common in mmdet pipelines)
            kwargs.pop("rescale", None)
            return self.forward_test(imgs, img_metas, **kwargs)

    def __call__(self, *args, **kwargs):
        """Override SimNN.Module.__call__ to properly dispatch"""
        return self.forward(*args, **kwargs)

    # ============================================================
    # Training and Validation Steps
    # ============================================================

    def train_step(self, data_dict, optimizer=None):
        """
        Single iteration step during training

        This method defines one training iteration for ttsim simulation.

        Args:
            data_dict (dict): Output from dataloader, typically containing:
                - imgs (SimTensor): Input images
                - img_metas (list[dict]): Meta information
                - gt_* : Ground truth annotations (as SimTensors or numpy arrays)
            optimizer: Unused in ttsim context

        Returns:
            dict: Dictionary containing:
                - loss (SimTensor): Total loss with .data computed
                - log_vars (dict): Variables to send to logger (extracted from .data)
                - num_samples (int): Batch size for averaging logs
        """
        # Forward pass
        losses = self(**data_dict)

        # Parse outputs (handle both dict and tuple returns)
        if isinstance(losses, dict):
            loss = losses.get("loss")
            log_vars = losses.get("log_vars", {})
            num_samples = losses.get("num_samples", len(data_dict.get("img_metas", [])))
        else:
            # Tuple format: (loss, log_vars, num_samples)
            loss, log_vars, num_samples = losses

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

    def val_step(self, data_dict, optimizer=None):
        """
        Single iteration step during validation

        This method runs during validation for ttsim simulation.

        Args:
            data_dict (dict): Output from dataloader
            optimizer: Unused in ttsim context

        Returns:
            dict: Dictionary containing:
                - loss (SimTensor): Validation loss with .data computed
                - log_vars (dict): Variables to send to logger
                - num_samples (int): Batch size for averaging logs
        """
        # Forward pass
        losses = self(**data_dict)

        # Parse outputs (handle both dict and tuple returns)
        if isinstance(losses, dict):
            loss = losses.get("loss")
            log_vars = losses.get("log_vars", {})
            num_samples = losses.get("num_samples", len(data_dict.get("img_metas", [])))
        else:
            # Tuple format: (loss, log_vars, num_samples)
            loss, log_vars, num_samples = losses

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

    # ============================================================
    # Visualization
    # ============================================================

    def show_result(self, img, result, **kwargs):
        """
        Visualize detection/segmentation results

        Args:
            img (SimTensor or ndarray): Input image
            result (dict or list): Detection/segmentation results
            **kwargs: Additional visualization arguments

        Returns:
            ndarray: Visualized image (placeholder implementation)
        """
        # Placeholder - subclasses should implement actual visualization
        self.logger.warning("show_result() not implemented, returning None")
        return None

    def analytical_param_count(self, lvl):
        """
        Calculate total number of trainable parameters
        Should be overridden by subclasses to count their specific components

        Args:
            lvl: Detail level (unused in base implementation)

        Returns:
            int: Total parameter count (0 in base class)
        """
        # Base class has no parameters
        # Subclasses should override to count backbone, neck, head params
        return 0
