# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List, Union
import cv2
from PIL import Image
import glob
import torchvision.transforms as transforms


class YUVVideoDataset(Dataset):
    """
    YUV video dataset class.

    Defined in: YUVVideoDataset class, this file, lines 35-350.

    Reads raw video files in YUV format and converts them to RGB tensors.
    """

    def __init__(
        self,
        video_dir: str,
        width: int,
        height: int,
        pixel_format: str = 'yuv420p',
        bit_depth: int = 8,
        num_frames: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        resize: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        value_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        """
        Initialize the YUV video dataset.

        Defined in: YUVVideoDataset.__init__ method, this file, lines 52-105.

        Args:
            video_dir: directory containing YUV video files; all .yuv files in it are loaded automatically
            width: video width (pixels)
            height: video height (pixels)
            pixel_format: pixel format; supports 'yuv420p', 'yuv420p10le', 'yuv444p', etc.
            bit_depth: bit depth, 8 or 10
            num_frames: read the first N frames; None means read all frames
            crop_size: crop size (crop_h, crop_w); None means no cropping
            crop_position: crop start position (y, x); None means starting from the top-left corner (0, 0)
            normalize: whether to normalize pixel values
            value_range: value range after normalization, default (-1.0, 1.0)

        Example:
            >>> dataset = YUVVideoDataset(
            ...     video_dir='/path/to/yuv/videos',
            ...     width=1920,
            ...     height=1080,
            ...     pixel_format='yuv420p',
            ...     num_frames=100,
            ...     crop_size=(256, 256)
            ... )
        """
        self.video_dir = video_dir
        self.video_paths = self._scan_yuv_files(video_dir)
        self.width = width
        self.height = height
        self.pixel_format = pixel_format.lower()
        self.bit_depth = bit_depth
        self.num_frames = num_frames
        self.crop_size = crop_size
        if crop_size is not None:
            self.transforms=transforms.CenterCrop(crop_size)
        self.resize = resize
        if resize is not None:
            self.transforms=transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC)
        self.normalize = normalize
        self.value_range = value_range

        # Compute the YUV data size per frame (in bytes)
        self._calculate_frame_size()

        # Validate parameters
        self._validate_params()

    def _scan_yuv_files(self, video_dir: str) -> List[str]:
        """
        Scan all .yuv files under the given directory.

        Defined in: YUVVideoDataset._scan_yuv_files method, this file, lines 98-125.

        Args:
            video_dir: directory containing YUV video files

        Returns:
            List of YUV file paths (sorted by file name)

        Example:
            >>> dataset = YUVVideoDataset('/data/videos', 1920, 1080)
            >>> # Suppose /data/videos contains:
            >>> #   - video1.yuv
            >>> #   - video2.yuv
            >>> #   - subdir/video3.yuv
            >>> print(dataset.video_paths)
            ['/data/videos/video1.yuv', '/data/videos/video2.yuv']
        """
        if not os.path.isdir(video_dir):
            raise ValueError(f"The provided path is not a valid directory: {video_dir}")

        # Use glob to match all .yuv files (does not recurse into subdirectories)
        pattern = os.path.join(video_dir, "*.yuv")
        yuv_files = glob.glob(pattern)

        # Sort by file name to ensure a consistent order
        yuv_files = sorted(yuv_files)

        if len(yuv_files) == 0:
            raise ValueError(f"No .yuv files found under directory {video_dir}")

        print(f"Found {len(yuv_files)} YUV video files:")
        for f in yuv_files[:5]:  # only show the first 5
            print(f"  - {os.path.basename(f)}")
        if len(yuv_files) > 5:
            print(f"  ... and {len(yuv_files) - 5} more files")

        return yuv_files

    def _calculate_frame_size(self):
        """
        Compute the YUV data size per frame.

        Defined in: YUVVideoDataset._calculate_frame_size method, this file, lines 107-140.
        """
        # Bytes per pixel
        bytes_per_pixel = 2 if self.bit_depth > 8 else 1

        if self.pixel_format in ['yuv420p', '420p', 'i420', 'yv12']:
            # YUV 420: Y = W*H, U = W*H/4, V = W*H/4
            # Total size = 1.5 * W * H * bytes_per_pixel
            self.frame_size = int(self.width * self.height * 1.5 * bytes_per_pixel)
            self.uv_width = self.width // 2
            self.uv_height = self.height // 2
            self.subsample_ratio = 2
        elif self.pixel_format in ['yuv444p', '444p']:
            # YUV 444: Y = U = V = W*H
            # Total size = 3 * W * H * bytes_per_pixel
            self.frame_size = int(self.width * self.height * 3 * bytes_per_pixel)
            self.uv_width = self.width
            self.uv_height = self.height
            self.subsample_ratio = 1
        elif self.pixel_format in ['yuv422p', '422p']:
            # YUV 422: Y = W*H, U = V = W*H/2
            # Total size = 2 * W * H * bytes_per_pixel
            self.frame_size = int(self.width * self.height * 2 * bytes_per_pixel)
            self.uv_width = self.width // 2
            self.uv_height = self.height
            self.subsample_ratio = 2
        else:
            raise ValueError(f"Unsupported pixel format: {self.pixel_format}")

    def _validate_params(self):
        """
        Validate parameters.

        Defined in: YUVVideoDataset._validate_params method, this file, lines 142-165.
        """
        # Validate crop parameters
        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            if crop_h > self.height or crop_w > self.width:
                raise ValueError(
                    f"Crop size ({crop_h}, {crop_w}) must not exceed the video size ({self.height}, {self.width})"
                )


        # Validate bit depth
        if self.bit_depth not in [8, 10]:
            raise ValueError(f"Only 8 or 10 bit depth is supported, got {self.bit_depth}")

    def _get_total_frames(self, video_path: str) -> int:
        """
        Get the total number of frames of a video file.

        Defined in: YUVVideoDataset._get_total_frames method, this file, lines 167-185.

        Args:
            video_path: YUV file path

        Returns:
            Total number of frames
        """
        file_size = os.path.getsize(video_path)
        total_frames = file_size // self.frame_size
        return total_frames

    def _read_yuv_frame(self, f, frame_idx: int) -> np.ndarray:
        """
        Read a single YUV frame and convert it to RGB.

        Defined in: YUVVideoDataset._read_yuv_frame method, this file, lines 187-260.

        Args:
            f: file object
            frame_idx: frame index

        Returns:
            RGB image, shape (H, W, 3), value range [0, 255]

        Example:
            Input: YUV 420p 8bit frame data:
                Y: 1920x1080 bytes
                U: 960x540 bytes
                V: 960x540 bytes
            Output:
                RGB: (1080, 1920, 3) numpy array, dtype=uint8
        """
        # Seek to the specified frame
        offset = frame_idx * self.frame_size
        f.seek(offset)

        # Determine the data type
        dtype = np.uint16 if self.bit_depth > 8 else np.uint8

        # Read the Y component
        y_size = self.width * self.height
        Y = np.fromfile(f, dtype=dtype, count=y_size).reshape((self.height, self.width))

        # Read the U and V components
        uv_size = self.uv_width * self.uv_height
        U = np.fromfile(f, dtype=dtype, count=uv_size).reshape((self.uv_height, self.uv_width))
        V = np.fromfile(f, dtype=dtype, count=uv_size).reshape((self.uv_height, self.uv_width))

        # Convert 10bit data to 8bit
        if self.bit_depth > 8:
            Y = (Y / (2 ** self.bit_depth - 1) * 255).astype(np.uint8)
            U = (U / (2 ** self.bit_depth - 1) * 255).astype(np.uint8)
            V = (V / (2 ** self.bit_depth - 1) * 255).astype(np.uint8)

        # Upsample U and V to the same size as Y (for the 420 format)
        if self.subsample_ratio > 1:
            U = cv2.resize(U, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            V = cv2.resize(V, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # Merge YUV
        yuv = np.stack([Y, U, V], axis=2)

        # YUV to RGB
        # OpenCV uses the YCrCb format, so the U/V order must be adjusted
        ycrcb = np.stack([Y, V, U], axis=2)  # YCrCb format
        rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        return rgb

    def _normalize(self, frames: np.ndarray) -> np.ndarray:
        """
        Normalize frame data.

        Defined in: YUVVideoDataset._normalize method, this file, lines 297-320.

        Args:
            frames: input frames, shape (T, H, W, 3), value range [0, 255]

        Returns:
            Normalized frames, shape (T, H, W, 3)
        """
        if not self.normalize:
            return frames

        # First normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0

        # Then map to the target range
        min_val, max_val = self.value_range
        frames = frames * (max_val - min_val) + min_val

        return frames

    def __len__(self) -> int:
        """
        Return the dataset size.

        Defined in: YUVVideoDataset.__len__ method, this file, lines 322-330.
        """
        return len(self.video_paths)

    def __getitem__(self, idx: int) :
        """
        Get a single video sample.

        Defined in: YUVVideoDataset.__getitem__ method, this file, lines 332-400.

        Args:
            idx: sample index

        Returns:
            Video tensor, shape (C, T, H, W)

        Example:
            >>> dataset = YUVVideoDataset(['video.yuv'], 1920, 1080, num_frames=10)
            >>> video = dataset[0]
            >>> print(video.shape)  # torch.Size([3, 10, 1080, 1920])
            >>> print(video.dtype)  # torch.float32
            >>> print(video.min(), video.max())  # depends on the normalize parameter, e.g. -1.0 1.0
        """
        video_path = self.video_paths[idx]

        # Get the total number of frames
        total_frames = self._get_total_frames(video_path)

        # Determine the number of frames to read
        if self.num_frames is not None:
            num_frames = min(self.num_frames, total_frames)
        else:
            num_frames = total_frames

        # Read frames
        frames = []
        with open(video_path, 'rb') as f:
            for frame_idx in range(num_frames):
                # Read YUV and convert to RGB
                rgb_frame = self._read_yuv_frame(f, frame_idx)

                frames.append(rgb_frame)

        # Convert to a numpy array: (T, H, W, 3)
        frames = np.stack(frames, axis=0)

        # Normalize
        frames = self._normalize(frames)
        # Convert to a Tensor: (T, H, W, 3) -> (C, T, H, W)
        frames_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()

        if self.crop_size is not None:
            frames_tensor=self.transforms(frames_tensor)
        if self.resize is not None:
            frames_tensor=self.transforms(frames_tensor)

        return {"frames": frames_tensor,"video_path": video_path}
