# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
import ttsim.front.ttnn as ttnn
from abc import abstractmethod

class BaseInstance3DBoxes(object):
    def __init__(self, tensor, box_dim=9, with_yaw=True, origin=(0.5, 0.5, 0)):
        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # 0 as a fake yaw and set with_yaw to False.
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = ttnn.concat(tensor, fake_rot, axis=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = ttnn.clone(tensor)

        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def volume(self):
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self):
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        return self.tensor[:, 6]

    @property
    def height(self):
        return self.tensor[:, 5]

    @property
    def top_height(self):
        return self.bottom_height + self.height

    @property
    def bottom_height(self):
        return self.tensor[:, 2]

    @property
    def center(self):
        return self.bottom_center

    @property
    def bottom_center(self):
        return self.tensor[:, :3]

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box."""

    @property
    def corners(self):
        """torch.Tensor: a tensor with 8 corners of each box."""

    @abstractmethod
    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or \
        rotation matrix.

        Args:
            angle (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.
        """

    @abstractmethod
    def flip(self, bev_direction="horizontal"):
        """Flip the boxes in BEV along given BEV direction."""

    def in_range_3d(self, box_range):
        in_range_flags = (
            (self.tensor[:, 0] > box_range[0])
            & (self.tensor[:, 1] > box_range[1])
            & (self.tensor[:, 2] > box_range[2])
            & (self.tensor[:, 0] < box_range[3])
            & (self.tensor[:, 1] < box_range[4])
            & (self.tensor[:, 2] < box_range[5])
        )
        return in_range_flags

    @abstractmethod
    def in_range_bev(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): The range of box
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: Indicating whether each box is inside \
                the reference range.
        """

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type \
                in the `dst` mode.
        """

    def scale(self, scale_factor):
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor

    def nonempty(self, threshold: float = 0.0):
        box = self.tensor
        size_x = box[..., 3]
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = (size_x > threshold) & (size_y > threshold) & (size_z > threshold)
        return keep

    def __getitem__(self, item):
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1), box_dim=self.box_dim, with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, f"Indexing on Boxes with {item} failed to return a matrix!"
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def __len__(self):
        return self.tensor.shape[0]

    def __repr__(self):
        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"

    def to(self, device):
        original_type = type(self)
        return original_type(self.tensor.to(device), box_dim=self.box_dim, with_yaw=self.with_yaw)

    def clone(self):
        original_type = type(self)
        return original_type(self.tensor.clone(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    @property
    def device(self):
        return self.tensor.device

    def __iter__(self):
        yield from self.tensor


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    YAW_AXIS = 2
