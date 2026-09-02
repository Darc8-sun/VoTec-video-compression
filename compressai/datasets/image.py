# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os

from torch.utils.data import Dataset

import pdb
import random


from abc import ABC
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg
import cv2
from PIL import Image
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
from glob import glob
import torch
from torchvision import transforms

# MEAN = torch.tensor([0.485, 0.456, 0.406]).mean().unsqueeze(0)
MEAN = torch.tensor([0.4141, 0.4252, 0.4050])
STD = torch.tensor([0.17759174, 0.18816505, 0.19405709])


def denorm_img(img):
    if len(img.shape) == 3:
        img = np.transpose(img, [1, 2, 0])  # torch [C,H,W]
        MEAN = torch.tensor([0.4141, 0.4252, 0.404]).mean().unsqueeze(0)
        STD = torch.tensor([0.229, 0.224, 0.225]).mean().unsqueeze(0)
    # elif len(img.shape) == 2:
    #     mean = np.mean([118.93, 113.97, 102.60])
    #     std = np.mean([69.85, 68.81, 72.45])

    return img * STD + MEAN


def resize_tensor2tensor(tensor):
    tensor = torch.squeeze(tensor).cpu().detach().numpy()
    tensor = np.moveaxis(tensor, 0, -1)
    tensor = cv2.resize(tensor, (128, 128))
    tensor = tensor.reshape([1, 3, 128, 128])
    tensor = torch.tensor(tensor, device='cuda:0')
    return tensor


class StereoMatchingDataset(ABC, VisionDataset):
    """Base interface for Stereo matching datasets"""

    _has_built_in_disparity_mask = False

    def __init__(self, root: str, transforms: Optional[Callable] = None):
        """
        Args:
            root(str): Root directory of the dataset.
            transforms(callable, optional): A function/transform that takes in Tuples of
                (images, disparities, valid_masks) and returns a transformed version of each of them.
                images is a Tuple of (``PIL.Image``, ``PIL.Image``)
                disparities is a Tuple of (``np.ndarray``, ``np.ndarray``) with shape (1, H, W)
                valid_masks is a Tuple of (``np.ndarray``, ``np.ndarray``) with shape (H, W)
                In some cases, when a dataset does not provide disparities, the ``disparities`` and
                ``valid_masks`` can be Tuples containing None values.
                For training splits generally the datasets provide a minimal guarantee of
                images: (``PIL.Image``, ``PIL.Image``)
                disparities: (``np.ndarray``, ``None``) with shape (1, H, W)
                Optionally, based on the dataset, it can return a ``mask`` as well:
                valid_masks: (``np.ndarray | None``, ``None``) with shape (H, W)
                For some test splits, the datasets provides outputs that look like:
                imgaes: (``PIL.Image``, ``PIL.Image``)
                disparities: (``None``, ``None``)
                Optionally, based on the dataset, it can return a ``mask`` as well:
                valid_masks: (``None``, ``None``)
        """
        super().__init__(root=root)
        self.transforms = transforms

        self._images = []  # type: ignore
        self._disparities = []  # type: ignore

    def _read_img(self, file_path):
        img = Image.open(file_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _scan_pairs(self, paths_left_pattern: str, paths_right_pattern: Optional[str] = None):

        left_paths = list(sorted(glob(paths_left_pattern)))

        right_paths: List[Union[None, str]]
        if paths_right_pattern:
            right_paths = list(sorted(glob(paths_right_pattern)))
        else:
            right_paths = list(None for _ in left_paths)

        if not left_paths:
            raise FileNotFoundError(f"Could not find any files matching the patterns: {paths_left_pattern}")

        if not right_paths:
            raise FileNotFoundError(f"Could not find any files matching the patterns: {paths_right_pattern}")

        if len(left_paths) != len(right_paths):
            raise ValueError(
                f"Found {len(left_paths)} left files but {len(right_paths)} right files using:\n "
                f"left pattern: {paths_left_pattern}\n"
                f"right pattern: {paths_right_pattern}\n"
            )

        paths = list((left, right) for left, right in zip(left_paths, right_paths))
        return paths

    def _read_disparity(self, file_path: str) -> Tuple:
        # function that returns a disparity map and an occlusion map
        pass

    def __getitem__(self, index: int, is_kitti=False) -> Tuple:
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 3 or 4-tuple with ``(img_left, img_right, disparity, Optional[valid_mask])`` where ``valid_mask``
                can be a numpy boolean mask of shape (H, W) if the dataset provides a file
                indicating which disparity pixels are valid. The disparity is a numpy array of
                shape (1, H, W) and the images are PIL images. ``disparity`` is None for
                datasets on which for ``split="test"`` the authors did not provide annotations.
        """
        img_left = self._read_img(self._images[index][0])
        img_right = self._read_img(self._images[index][1])
        if is_kitti:
            dsp_map_left, valid_mask_left = self._read_disparity(self._disparities[index // 2][0])
            dsp_map_right, valid_mask_right = self._read_disparity(self._disparities[index // 2][1])
        else:
            dsp_map_left, valid_mask_left = self._read_disparity(self._disparities[index][0])
            dsp_map_right, valid_mask_right = self._read_disparity(self._disparities[index][1])

        imgs = (img_left, img_right)
        dsp_maps = (dsp_map_left, dsp_map_right)
        valid_masks = (valid_mask_left, valid_mask_right)

        if self.transforms is not None:
            (
                imgs,
                dsp_maps,
                valid_masks,
            ) = self.transforms(imgs, dsp_maps, valid_masks)

        if self._has_built_in_disparity_mask or valid_masks[0] is not None:
            return imgs[0], imgs[1], dsp_maps[0], valid_masks[0]
        else:
            return imgs[0], imgs[1], dsp_maps[0]

    def __len__(self) -> int:
        return len(self._images)


class Kitti2015Stereo(StereoMatchingDataset):
    """
    KITTI dataset from the `2015 stereo evaluation benchmark <http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php>`_.

    The dataset is expected to have the following structure: ::

        root
            Kitti2015
                testing
                    image_2
                        img1.png
                        img2.png
                        ...
                    image_3
                        img1.png
                        img2.png
                        ...
                training
                    image_2
                        img1.png
                        img2.png
                        ...
                    image_3
                        img1.png
                        img2.png
                        ...
                    disp_occ_0
                        img1.png
                        img2.png
                        ...
                    disp_occ_1
                        img1.png
                        img2.png
                        ...
                    calib

    Args:
        root (string): Root directory where `Kitti2015` is located.
        split (string, optional): The dataset split of scenes, either "train" (default) or "test".
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
    """

    _has_built_in_disparity_mask = True

    def __init__(self, root: str, args=None, resize=False, no_norm=True, org=False, randomcrop=False,
                 split: str = "train", robust_parm=None):
        super().__init__(root)

        verify_str_arg(split, "split", valid_values=("train", "test"))

        root = Path(root) / "Kitti2015" / (split + "ing")
        left_img_pattern = str(root / "image_2" / "*.png")
        right_img_pattern = str(root / "image_3" / "*.png")
        self._images = self._scan_pairs(left_img_pattern, right_img_pattern)
        self.args = args
        self.resize = resize
        self.robust_parm = robust_parm
        self.is_train = split
        self.org = org
        self.random_crop = randomcrop

        if no_norm:
            MEAN = 0
            STD = 1
        else:
            MEAN = torch.tensor([0.4141, 0.4252, 0.4050])
            STD = torch.tensor([0.17759174, 0.18816505, 0.19405709])
        self.crop = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(args.picsize),
                                        transforms.Normalize(mean=MEAN, std=STD),
                                        transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])
        self.datatrans = transforms.Compose(
            [
                # ywz
                # transforms.Resize(self.homopic_size),
                # #
                # transforms.CenterCrop(self.homopic_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
        if split == "train":
            left_disparity_pattern = str(root / "disp_occ_0" / "*.png")
            right_disparity_pattern = str(root / "disp_occ_1" / "*.png")
            self._disparities = self._scan_pairs(left_disparity_pattern, right_disparity_pattern)
        else:
            self._disparities = list((None, None) for _ in self._images)

    def _read_disparity(self, file_path: str) -> Tuple:
        # test split has no disparity maps
        if file_path is None:
            return None, None

        disparity_map = np.asarray(Image.open(file_path)) / 256.0
        # unsqueeze the disparity map into (C, H, W) format
        disparity_map = disparity_map[None, :, :]
        valid_mask = None
        return disparity_map, valid_mask

    def _getitem_(self, index: int) -> Tuple:
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 4-tuple with ``(img_left, img_right, disparity, valid_mask)``.
            The disparity is a numpy array of shape (1, H, W) and the images are PIL images.
            ``valid_mask`` is implicitly ``None`` if the ``transforms`` parameter does not
            generate a valid mask.
            Both ``disparity`` and ``valid_mask`` are ``None`` if the dataset split is test.
        """
        return super().__getitem__(index, is_kitti=True)

    def __getitem__(self, index):
        img1, img2, disp_left, _ = self._getitem_(index)
        if self.robust_parm != None:
            img2 = transforms.ColorJitter(brightness=self.robust_parm)(img2)
        img1, img2 = np.array(img1.convert("RGB")), np.array(img2.convert("RGB"))
        assert np.all(img1.shape[:2] == img2.shape[:2])
        img = np.concatenate([img1, img2], axis=2)
        if self.random_crop:
            img_a, img_b = np.split(img, 2, 2)
            img_a = self.crop(img_a / 255)
            img_b = self.crop(img_b / 255)
            return img_a.float(), img_b.float()
        if self.resize or self.org:
            height, weight = img.shape[:2]
            H = height - height % 64
            W = weight - weight % 64
            h_begin, w_begin = (height - H) // 2, (weight - W) // 2
            img = img[h_begin:h_begin + H, w_begin:w_begin + W]
            if self.is_train == 'test':
                pass
            else:
                disp_left = disp_left[:, h_begin:h_begin + H, w_begin:w_begin + W]
        else:
            height, weight = img.shape[:2]
            h_begin, w_begin = (height - self.args.patch_size[0]) // 2, (weight - self.args.patch_size[1]) // 2
            img = img[h_begin:h_begin + self.args.patch_size[0], w_begin:w_begin + self.args.patch_size[1]]
            if self.is_train == 'test':
                pass
            else:
                disp_left = disp_left[:, h_begin:h_begin + self.args.patch_size[0],
                            w_begin:w_begin + self.args.patch_size[1]]
        img_a, img_b = np.split(img, 2, 2)
        img_a = self.datatrans(img_a / 255)
        img_b = self.datatrans(img_b / 255)
        if self.resize:
            img_a = cv2.resize(img_a, (self.args.patch_size[0], self.args.patch_size[1]))
            img_b = cv2.resize(img_b, (self.args.patch_size[0], self.args.patch_size[1]))
        if self.is_train == 'train':
            return img_a.float(), img_b.float(), torch.tensor(disp_left, dtype=torch.float)
        else:
            return img_a.float(), img_b.float()


class InStereo2k(StereoMatchingDataset):
    """`InStereo2k <https://github.com/YuhuaXu/StereoDataset>`_ dataset.

    The dataset is expected to have the following structre: ::

        root
            InStereo2k
                train
                    scene1
                        left.png
                        right.png
                        left_disp.png
                        right_disp.png
                        ...
                    scene2
                    ...
                test
                    scene1
                        left.png
                        right.png
                        left_disp.png
                        right_disp.png
                        ...
                    scene2
                    ...

    Args:
        root (string): Root directory where InStereo2k is located.
        split (string): Either "train" or "test".
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
    """

    def __init__(self, root: str, args=None, is_train=True, for_homo_train=False, change=False, foursplit=False,
                 no_norm=False, resize=False, org=False, stero_matching=False, random_crop=False, split: str = "train",
                 robust_parm=None, scalecrop=None,single_image=False):
        super().__init__(root)

        root = Path(root) / "InStereo2k" / split

        verify_str_arg(split, "split", valid_values=("train", "test"))

        left_img_pattern = str(root / "*" / "left.png")
        right_img_pattern = str(root / "*" / "right.png")
        self.no_norm = no_norm
        self.org = org
        self.random_crop = random_crop
        self.stero_matching = stero_matching
        self.resize = resize
        if no_norm:
            MEAN = 0
            STD = 1
        else:
            MEAN = torch.tensor([0.4141, 0.4252, 0.4050])
            STD = torch.tensor([0.17759174, 0.18816505, 0.19405709])
        self._images = self._scan_pairs(left_img_pattern, right_img_pattern)
        self.change = change
        self.single_image=single_image
        self.scalecrop = scalecrop
        left_disparity_pattern = str(root / "*" / "left_disp.png")
        right_disparity_pattern = str(root / "*" / "right_disp.png")
        self._disparities = self._scan_pairs(left_disparity_pattern, right_disparity_pattern)
        self.args = args
        self.foursplit = foursplit
        self.is_train = is_train
        self.for_homo_train = for_homo_train
        self.crop = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(args.picsize),
                                        transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])
        self.robust_parm = robust_parm
        self.homotransforms = transforms.Compose(
            [
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                # ywz
                transforms.Resize(args.picsize),
                #
                transforms.CenterCrop(args.picsize),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
        self.homotransforms_fortest = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def _read_disparity(self, file_path: str) -> Tuple:
        disparity_map = np.asarray(Image.open(file_path), dtype=np.float32)
        # unsqueeze disparity to (C, H, W)
        disparity_map = disparity_map[None, :, :] / 1024.0
        valid_mask = None
        return disparity_map, valid_mask

    def _getmeanvar(self, index):
        img1, img2 = self._getitem_(index)[:2]
        img1, img2 = np.array(img1.convert("RGB")) / 255, np.array(img2.convert("RGB")) / 255
        mean = img1.mean(axis=(0, 1)) + img2.mean(axis=(0, 1))
        var = img1.std(axis=(0, 1)) + img2.std(axis=(0, 1))
        return mean, var

    def _getitem_(self, index: int) -> Tuple:
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 3-tuple with ``(img_left, img_right, disparity)``.
            The disparity is a numpy array of shape (1, H, W) and the images are PIL images.
            If a ``valid_mask`` is generated within the ``transforms`` parameter,
            a 4-tuple with ``(img_left, img_right, disparity, valid_mask)`` is returned.
        """
        return super().__getitem__(index)
    def __getitem__(self, index):
        if self.single_image:
            index_a=index//2
            index_b=index%2
            img=self._getitem_(index_a)[index_b]
            img = np.array(img.convert("RGB"))
            if self.resize or self.org:
                height, weight = img.shape[:2]
                H = height - height % 64
                W = weight - weight % 64
                h_begin, w_begin = (height - H) // 2, (weight - W) // 2
                img = img[h_begin:h_begin + H, w_begin:w_begin + W]
            else:
                if self.random_crop:
                    img_a, img_b = np.split(img, 2, 2)
                    img_a = self.crop(img_a / 255)
                    img_b = self.crop(img_b / 255)
                    return img_a.float(), img_b.float()
                else:
                    height, weight = img.shape[:2]
                    h_begin, w_begin = (height - self.args.patch_size[0]) // 2, (weight - self.args.patch_size[1]) // 2
                    img = img[h_begin:h_begin + self.args.patch_size[0], w_begin:w_begin + self.args.patch_size[1]]
            img = self.homotransforms_fortest(img / 255)
            if self.resize:
                img = cv2.resize(img, (self.args.patch_size[0], self.args.patch_size[1]))
            return img.float()
        if self.stero_matching:
            img1, img2, disp_left = self._getitem_(index)
            if self.robust_parm != None:
                img2 = transforms.ColorJitter(brightness=self.robust_parm)(img2)
            img1, img2 = np.array(img1.convert("RGB")), np.array(img2.convert("RGB"))
            assert np.all(img1.shape[:2] == img2.shape[:2])
            img = np.concatenate([img1, img2], axis=2)
            if self.resize or self.org:
                height, weight = img.shape[:2]
                H = height - height % 64
                W = weight - weight % 64
                h_begin, w_begin = (height - H) // 2, (weight - W) // 2
                img = img[h_begin:h_begin + H, w_begin:w_begin + W]
                disp_left = disp_left[:, h_begin:h_begin + H, w_begin:w_begin + W]
            else:
                if self.random_crop:
                    img_a, img_b = np.split(img, 2, 2)
                    img_a = self.crop(img_a / 255)
                    img_b = self.crop(img_b / 255)
                    return img_a.float(), img_b.float()
                else:
                    height, weight = img.shape[:2]
                    h_begin, w_begin = (height - self.args.patch_size[0]) // 2, (weight - self.args.patch_size[1]) // 2
                    img = img[h_begin:h_begin + self.args.patch_size[0], w_begin:w_begin + self.args.patch_size[1]]

                    disp_left = disp_left[:, h_begin:h_begin + self.args.patch_size[0],
                                w_begin:w_begin + self.args.patch_size[1]]
            img_a, img_b = np.split(img, 2, 2)
            if self.scalecrop != None:
                height, weight = img_b.shape[:2]
                img_b[int(self.scalecrop * height):, int(self.scalecrop * weight):] = 0
            img_a = self.homotransforms_fortest(img_a / 255)
            img_b = self.homotransforms_fortest(img_b / 255)
            if self.resize:
                img_a = cv2.resize(img_a, (self.args.patch_size[0], self.args.patch_size[1]))
                img_b = cv2.resize(img_b, (self.args.patch_size[0], self.args.patch_size[1]))
            return img_a.float(), img_b.float(), torch.tensor(disp_left, dtype=torch.float)
        if self.change:
            img2, img1 = self._getitem_(index)[:2]
        else:
            img1, img2 = self._getitem_(index)[:2]
        # img = np.concatenate([img1, img2], axis=2)
        # if self.is_train:
        #     # if self.args.randomcrop:
        #     #     import random
        #     #     height, weight = img.shape[:2]
        #     #     h_begin, w_begin = random.randint(0, height - self.args.patch_size[0]), random.randint(0, weight -
        #     #                                                                                            self.args.patch_size[                                                                                                       1])
        #     #     img = img[h_begin:h_begin + self.args.patch_size[0], w_begin:w_begin + self.args.patch_size[1]]
        #     # elif self.args.centercrop:
        #     height, weight = img.shape[:2]
        #     h_begin, w_begin = (height - self.args.patch_size[0]) // 2, (weight - self.args.patch_size[1]) // 2
        #     img = img[h_begin:h_begin + self.args.patch_size[0], w_begin:w_begin + self.args.patch_size[1]]
        if not self.for_homo_train:
            img1, img2 = np.array(img1.convert("RGB")), np.array(img2.convert("RGB"))
            assert np.all(img1.shape[:2] == img2.shape[:2])
            img = np.concatenate([img1, img2], axis=2)
            if self.resize:
                pass
            else:
                height, weight = img.shape[:2]
                h_begin, w_begin = (height - self.args.patch_size[0]) // 2, (weight - self.args.patch_size[1]) // 2
                img = img[h_begin:h_begin + self.args.patch_size[0], w_begin:w_begin + self.args.patch_size[1]]
            img_a, img_b = np.split(img, 2, 2)
            if self.resize:
                img_a = cv2.resize(img_a, (self.args.patch_size[0], self.args.patch_size[1]))
                img_b = cv2.resize(img_b, (self.args.patch_size[0], self.args.patch_size[1]))
            img_a, img_b = np.split(img, 2, 2)
            # img_homo_a = cv2.resize(img_a, (self.args.picsize, self.args.picsize))
            # img_homo_b = cv2.resize(img_b, (self.args.picsize, self.args.picsize))
            img_homo_a = self.homotransforms_fortest(img_a / 255)
            img_homo_b = self.homotransforms_fortest(img_b / 255)
            img_a = self.homotransforms_fortest(img_a / 255)
            img_b = self.homotransforms_fortest(img_b / 255)
            img = torch.cat([img_a, img_b], dim=0)
            if self.args.getpatch:
                patch_a, patch_b, corners = Getpatch(img_homo_a, img_homo_b, pic_size=self.args.picsize,
                                                     patch_size=self.args.patchsize,
                                                     rho=self.args.rho, m=self.args.m, center=False)
                patch_a = torch.mean(patch_a, dim=0, keepdim=True)
                patch_b = torch.mean(patch_b, dim=0, keepdim=True)
                img_patch = torch.cat([patch_a, patch_b], dim=0)
                # img = np.transpose(np.array(img), (2, 0, 1))
                return img.float(), img_patch.float(), corners.float()
            elif self.args.getcorner:
                corners = Getpatch_feature_domain(img, pic_size=self.args.patch_size[0] // 2, rho=self.args.rho,
                                                  patch_size=self.args.patchsize, m=self.args.m)
                return img.float(), corners.float()
            else:
                return img.float()
        else:
            img1, img2 = np.array(img1.convert("RGB")), np.array(img2.convert("RGB"))
            assert np.all(img1.shape[:2] == img2.shape[:2])
            img = np.concatenate([img1, img2], axis=2)
            if self.resize:
                pass
            else:
                height, weight = img.shape[:2]
                h_begin, w_begin = (height - self.args.patch_size[0]) // 2, (weight - self.args.patch_size[1]) // 2
                img = img[h_begin:h_begin + self.args.patch_size[0], w_begin:w_begin + self.args.patch_size[1]]
            img_a, img_b = np.split(img, 2, 2)
            if self.resize:
                img_a = cv2.resize(img_a, (self.args.patch_size[0], self.args.patch_size[1]))
                img_b = cv2.resize(img_b, (self.args.patch_size[0], self.args.patch_size[1]))
            img_homo_a = cv2.resize(img_a, (self.args.picsize, self.args.picsize))
            img_homo_b = cv2.resize(img_b, (self.args.picsize, self.args.picsize))
            img_homo_a = self.homotransforms_fortest(img_homo_a / 255)
            img_homo_b = self.homotransforms_fortest(img_homo_b / 255)
            if self.foursplit:
                patch_a1, patch_b1, patch_a2, patch_b2, patch_a3, patch_b3, patch_a4, patch_b4, corners1, corners2, corners3, corners4 = GETFOURPATCH(
                    img_homo_a, img_homo_b, self.args.patchsize)
                patch_a1 = torch.mean(patch_a1, dim=0, keepdim=True)
                patch_b1 = torch.mean(patch_b1, dim=0, keepdim=True)
                patch1 = torch.cat([patch_a1, patch_b1], dim=0).float()
                patch_a2 = torch.mean(patch_a2, dim=0, keepdim=True)
                patch_b2 = torch.mean(patch_b2, dim=0, keepdim=True)
                patch2 = torch.cat([patch_a2, patch_b2], dim=0).float()
                patch_a3 = torch.mean(patch_a3, dim=0, keepdim=True)
                patch_b3 = torch.mean(patch_b3, dim=0, keepdim=True)
                patch3 = torch.cat([patch_a3, patch_b3], dim=0).float()
                patch_a4 = torch.mean(patch_a4, dim=0, keepdim=True)
                patch_b4 = torch.mean(patch_b4, dim=0, keepdim=True)
                patch4 = torch.cat([patch_a4, patch_b4], dim=0).float()
            else:
                patch_a, patch_b, corners = Getpatch(img_homo_a, img_homo_b, pic_size=self.args.picsize,
                                                     patch_size=self.args.patchsize,
                                                     rho=self.args.rho, m=self.args.m, center=False)
                patch_a = torch.mean(patch_a, dim=0, keepdim=True)
                patch_b = torch.mean(patch_b, dim=0, keepdim=True)
                img_patch = torch.cat([patch_a, patch_b], dim=0)
            img_homo_a = torch.mean(img_homo_a, dim=0, keepdim=True)
            img_homo_b = torch.mean(img_homo_b, dim=0, keepdim=True)
            img = torch.cat([img_homo_a, img_homo_b], dim=0)
            if self.foursplit:
                return img.float(), [patch1, patch2, patch3, patch4], [corners1, corners2, corners3, corners4]
            else:
                return img.float(), img_patch.float(), corners.float()

class ImageFolderfromTxtwithSegLabel(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, txt_dir, args, transform=None, is_train=True, split="train"):
        pdb.set_trace()
        f = open(txt_dir, 'r')
        samples = f.readlines()

        self.samples = [os.path.join(args.dataset_dir, sample.strip()) for sample in samples]
        self.samples = list(zip(self.samples[0:][::3], self.samples[1:][::3], self.samples[2:][::3]))

        if len(self.samples) == 0:
            raise RuntimeError(f'Invalid txt directory "{txt_dir}"')
        self.args = args
        self.transform = transform
        self.is_train = is_train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        if self.args.is_pair:
            img1 = np.array(Image.open(self.samples[index][0]).convert("RGB"))
            img2 = np.array(Image.open(self.samples[index][2]).convert("RGB"))
            assert np.all(img1.shape[:2] == img2.shape[:2])
            img = np.concatenate([img1, img2], axis=2)

        else:
            img1 = np.array(Image.open(self.samples[index][0]).convert("RGB"))
            img = img1

        mask = np.array(Image.open(self.samples[index][1]))
        mask = self._preprocess_mask(mask)

        if self.is_train:
            if self.args.randomcrop:
                import random
                height, weight = img.shape[:2]
                h_begin, w_begin = random.randint(0, height - self.args.patch_size[0]), random.randint(0, weight -
                                                                                                       self.args.patch_size[
                                                                                                           1])
                img = img[h_begin:h_begin + self.args.patch_size[0], w_begin:w_begin + self.args.patch_size[1]]
                mask = mask[h_begin:h_begin + self.args.patch_size[0], w_begin:w_begin + self.args.patch_size[1]]

            elif self.args.centercrop:
                height, weight = img.shape[:2]
                h_begin, w_begin = (height - self.args.patch_size[0]) // 2, (weight - self.args.patch_size[1]) // 2
                img = img[h_begin:h_begin + self.args.patch_size[0], w_begin:w_begin + self.args.patch_size[1]]
                mask = mask[h_begin:h_begin + self.args.patch_size[0], w_begin:w_begin + self.args.patch_size[1]]

        img = np.transpose(np.array(img.astype(np.float32)), (2, 0, 1)) / 255.
        mask = np.expand_dims(mask, 0)

        if self.transform:
            return self.transform(img)

        sample = dict(img=torch.from_numpy(img).float(), mask=torch.from_numpy(mask).float())
        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask >= 1.0] = 1.0
        return mask

    def __len__(self):
        return len(self.samples)


class ImageFolderfromTxt(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, txt_dir, args, transform=None, for_homo_train=False, is_train=True, split="train",
                 ori_sample=False):
        f = open(txt_dir, 'r')
        samples = f.readlines()

        self.samples = [os.path.join(args.dataset_dir, sample.strip()) for sample in samples]
        if args.is_pair:
            self.samples = list(zip(self.samples[0:][::2], self.samples[1:][::2]))

        if len(self.samples) == 0:
            raise RuntimeError(f'Invalid txt directory "{txt_dir}"')
        self.args = args
        self.transform = transform
        self.is_train = is_train
        self.ori_sample = ori_sample
        self.is_visualization = args.is_visualization
        self.for_homo_train = for_homo_train
        self.homotransforms = transforms.Compose(
            [
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                # ywz
                transforms.Resize(args.patch_size[0]),
                #
                transforms.CenterCrop(args.patch_size[0]),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
        self.homotransforms_fortest = transforms.Compose(
            [
                # ywz
                # transforms.Resize(self.homopic_size),
                # #
                # transforms.CenterCrop(self.homopic_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img1 = Image.open(self.samples[index][0])
        img2 = Image.open(self.samples[index][1])
        if not self.for_homo_train:
            img1, img2 = np.array(img1.convert("RGB")), np.array(img2.convert("RGB"))
            assert np.all(img1.shape[:2] == img2.shape[:2])
            img = np.concatenate([img1, img2], axis=2)
            height, weight = img.shape[:2]
            h_begin, w_begin = (height - self.args.patch_size[0]) // 2, (weight - self.args.patch_size[1]) // 2
            img = img[h_begin:h_begin + self.args.patch_size[0], w_begin:w_begin + self.args.patch_size[1]]
            img_a, img_b = np.split(img, 2, 2)
            img_homo_a = cv2.resize(img_a, (self.args.picsize, self.args.picsize))
            img_homo_b = cv2.resize(img_b, (self.args.picsize, self.args.picsize))
            img_homo_a = self.homotransforms_fortest(img_homo_a / 255)
            img_homo_b = self.homotransforms_fortest(img_homo_b / 255)
            if self.args.getpatch:
                patch_a, patch_b, corners = Getpatch(img_homo_a, img_homo_b, pic_size=self.args.picsize,
                                                     patch_size=self.args.patchsize,
                                                     rho=self.args.rho, m=self.args.m)
                patch_a = torch.mean(patch_a, dim=0, keepdim=True)
                patch_b = torch.mean(patch_b, dim=0, keepdim=True)
                img_patch = torch.cat([patch_a, patch_b], dim=0)
                img = np.transpose(np.array(img), (2, 0, 1))
                return torch.from_numpy(img).float() / 255, img_patch.float(), corners.float()
            elif self.args.getcorner:
                img = np.transpose(np.array(img), (2, 0, 1))
                corners = Getpatch_feature_domain(img, rho=self.args.rho, patch_size=self.args.patchsize, m=self.args.m)
                return torch.from_numpy(img).float() / 255, corners.float()
            else:
                img = np.transpose(np.array(img), (2, 0, 1))
                return torch.from_numpy(img).float() / 255
        else:
            self.args.getpatch = True
            self.args.getcorner = False
            img_a, img_b = self.homotransforms(img1), self.homotransforms(img2)
            img_a = torch.mean(img_a, dim=0, keepdim=True)
            img_b = torch.mean(img_b, dim=0, keepdim=True)
            patch_a, patch_b, corners = Getpatch(img_a, img_b, pic_size=self.args.picsize,
                                                 patch_size=self.args.patchsize,
                                                 rho=self.args.rho, m=self.args.m)
            img_patch = torch.cat([patch_a, patch_b], dim=0)
            img = torch.cat([img_a, img_b], dim=0)
            return img.float(), img_patch.float(), corners.float()

    def _getitem(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def getitem(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        if self.args.is_pair:
            img1 = np.array(Image.open(self.samples[index][0]).convert("RGB"))
            img2 = np.array(Image.open(self.samples[index][1]).convert("RGB"))
            assert np.all(img1.shape[:2] == img2.shape[:2])
            img = np.concatenate([img1, img2], axis=2)
        else:
            img = Image.open(self.samples[index]).convert("RGB")
            img = np.array(img)

        if self.args.crop:
            import random
            height, weight = img.shape[:2]
            h_begin, w_begin = random.randint(0, height - self.args.patch_size[0]), random.randint(0, weight -
                                                                                                   self.args.patch_size[
                                                                                                       1])
            img = img[h_begin:h_begin + self.args.patch_size[0], w_begin:w_begin + self.args.patch_size[1]]
        elif self.args.pad:
            height, weight = img.shape[:2]
            res_h, res_w = self.args.padding_size - height % self.args.padding_size, self.args.padding_size - weight % self.args.padding_size
            paddings = ((0, res_h), (0, res_w), (0, 0))
            img = np.pad(img, paddings, 'constant')

        img = np.transpose(np.array(img), (2, 0, 1))
        if self.transform:
            return self.transform(img)
        return torch.from_numpy(img).float()

    def __len__(self):
        return len(self.samples)


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class CropCityscapesArtefacts:
    """Crop Cityscapes images to remove artefacts"""

    def __init__(self):
        self.top = 64
        self.left = 128
        self.right = 128
        self.bottom = 256

    def __call__(self, image):
        """Crops a PIL image.
        Args:
            image (PIL.Image): Cityscapes image (or disparity map)
        Returns:
            PIL.Image: Cropped PIL Image
        """
        w, h = image.size
        assert w == 2048 and h == 1024, f'Expected (2048, 1024) image but got ({w}, {h}). Maybe the ordering of transforms is wrong?'
        # w, h = 1792, 704
        return image.crop((self.left, self.top, w - self.right, h - self.bottom))


class MinimalCrop:
    """
    Performs the minimal crop such that height and width are both divisible by min_div.
    """

    def __init__(self, min_div=16):
        self.min_div = min_div

    def __call__(self, image):
        w, h = image.size

        h_new = h - (h % self.min_div)
        w_new = w - (w % self.min_div)

        if h_new == 0 and w_new == 0:
            return image
        else:
            h_diff = h - h_new
            w_diff = w - w_new

            top = int(h_diff / 2)
            bottom = h_diff - top
            left = int(w_diff / 2)
            right = w_diff - left

            return image.crop((left, top, w - right, h - bottom))


class Cityscape(Dataset):
    """Dataset class for image compression datasets."""

    # /home/xzhangga/datasets/Instereo2K/train/
    def __init__(self, ds_type='train', ds_name='cityscapes', root="/data/intern6/",
                 crop_size=(256, 256), resize=False, **kwargs):
        """
        Args:
            name (str): name of dataset, template: ds_name#ds_type. No '#' in ds_name or ds_type allowed. ds_type in (train, eval, test).
            path (str): if given the dataset is loaded from path instead of by name.
            transforms (Transform): transforms to apply to image
            debug (bool, optional): If set to true, limits the list of files to 10. Defaults to False.
        """
        super().__init__()

        self.path = Path(f"{root}")
        self.ds_name = ds_name
        self.ds_type = ds_type
        if ds_type == "train":
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.RandomVerticalFlip(p=0.5), ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.left_image_list, self.right_image_list = self.get_files()
        # if ds_type == "train":
        #     self.crop = CropCityscapesArtefacts()
        # else:
        #     self.crop=transforms.CenterCrop((1024,1024))
        self.crop = CropCityscapesArtefacts()
        print(f'Loaded dataset {ds_name} from {self.path}. Found {len(self.left_image_list)} files.')

    def __len__(self):
        return len(self.left_image_list)

    def __getitem__(self, index):
        # self.index_count += 1
        image_list = [Image.open(self.left_image_list[index]).convert('RGB'),
                      Image.open(self.right_image_list[index]).convert('RGB')]
        if self.crop is not None:
            image_list = [self.crop(image) for image in image_list]
        frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
        frames = torch.chunk(self.transform(frames), 2)
        if random.random() < 0.5:
            frames = frames[::-1]
        return frames

    def get_files(self):
        left_image_list, right_image_list, disparity_list = [], [], []
        for left_image_path in self.path.glob(f'leftImg8bit_trainvaltest/leftImg8bit/{self.ds_type}/*/*.png'):
            left_image_list.append(str(left_image_path))
            right_image_list.append(str(left_image_path).replace("leftImg8bit", 'rightImg8bit'))
            disparity_list.append(str(left_image_path).replace("leftImg8bit", 'disparity'))

        return left_image_list, right_image_list


