# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

import random
import os

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset

def load_data_yield(loader):
    while True:
        yield from loader

def load_data_inpa(
    *,
    gt_path=None,
    mask_path=None,
    mask2_path=None,
    mask3_path=None,
    inpaint_mask_path=None,
    filter_mask_path=None,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    return_dataloader=False,
    return_dict=False,
    max_len=None,
    drop_last=True,
    conf=None,
    offset=0,
    ** kwargs
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """

    gt_dir = os.path.expanduser(gt_path)
    mask_dir = os.path.expanduser(mask_path)
    mask2_dir = os.path.expanduser(mask2_path)
    mask3_dir = os.path.expanduser(mask3_path)
    inpaint_mask_dir = os.path.expanduser(inpaint_mask_path)
    filter_mask_dir = os.path.expanduser(filter_mask_path)

    gt_paths = _list_image_files_recursively(gt_dir)
    mask_paths = _list_image_files_recursively(mask_dir)
    mask2_paths = _list_image_files_recursively(mask2_dir)
    mask3_paths = _list_image_files_recursively(mask3_dir)
    inpaint_mask_paths = _list_image_files_recursively(inpaint_mask_dir)
    filter_mask_paths = _list_image_files_recursively(filter_mask_dir)


    assert len(gt_paths) == len(mask_paths)

    classes = None
    if class_cond:
        raise NotImplementedError()

    dataset = ImageDatasetInpa(
        image_size,
        gt_paths=gt_paths,
        mask_paths=mask_paths,
        mask2_paths=mask2_paths,
        mask3_paths=mask3_paths,
        inpaint_mask_paths=inpaint_mask_paths,
        filter_mask_paths=filter_mask_paths,
        classes=classes,
        shard=0,
        num_shards=1,
        random_crop=random_crop,
        random_flip=random_flip,
        return_dict=return_dict,
        max_len=max_len,
        conf=conf,
        offset=offset
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=drop_last
        )

    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=drop_last
        )

    if return_dataloader:
        return loader
    else:
        return load_data_yield(loader)


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDatasetInpa(Dataset):
    def __init__(
        self,
        resolution,
        gt_paths,
        mask_paths,
        mask2_paths,
        mask3_paths,
        inpaint_mask_paths,
        filter_mask_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        return_dict=False,
        max_len=None,
        conf=None,
        offset=0
    ):
        super().__init__()
        self.resolution = resolution

        gt_paths = sorted(gt_paths)[offset:]
        mask_paths = sorted(mask_paths)[offset:]
        mask2_paths = sorted(mask2_paths)[offset:]
        mask3_paths = sorted(mask3_paths)[offset:]
        inpaint_mask_paths = sorted(inpaint_mask_paths)[offset:]
        filter_mask_paths = sorted(filter_mask_paths)[offset:]

        self.local_gts = gt_paths[shard:][::num_shards]
        self.local_masks = mask_paths[shard:][::num_shards]
        self.local_masks2 = mask2_paths[shard:][::num_shards]
        self.local_masks3 = mask3_paths[shard:][::num_shards]
        self.local_inpain_masks = inpaint_mask_paths[shard:][::num_shards]
        self.local_filter_masks = filter_mask_paths[shard:][::num_shards]

        self.local_classes = None if classes is None else classes[shard:][::num_shards]

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.return_dict = return_dict
        self.max_len = max_len

    def __len__(self):
        if self.max_len is not None:
            return self.max_len

        return len(self.local_gts)

    def __getitem__(self, idx):
        gt_path = self.local_gts[idx]
        pil_gt = self.imread(gt_path)

        mask_path = self.local_masks[idx]
        pil_mask = self.imread(mask_path)

        mask2_path = self.local_masks2[idx]
        pil_mask2 = self.imread(mask2_path)

        mask3_path = self.local_masks3[idx]
        pil_mask3 = self.imread(mask3_path)

        inpaint_mask_path = self.local_inpain_masks[idx]
        pil__inpaint_mask = self.imread(inpaint_mask_path)
        
        filter_mask_path = self.local_filter_masks[idx]
        pil__filter_mask = self.imread(filter_mask_path)

        if self.random_crop:
            raise NotImplementedError()
        else:
            arr_gt = center_crop_arr(pil_gt, self.resolution)
            arr_mask = center_crop_arr(pil_mask, self.resolution)
            arr_mask2 = center_crop_arr(pil_mask2, self.resolution)
            arr_mask3 = center_crop_arr(pil_mask3, self.resolution)
            arr_inpaint_mask = center_crop_arr(pil__inpaint_mask, self.resolution)
            arr_filter_mask = center_crop_arr(pil__filter_mask, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr_gt = arr_gt[:, ::-1]
            arr_mask = arr_mask[:, ::-1]
            arr_mask2 = arr_mask2[:, ::-1]
            arr_mask3 = arr_mask3[:, ::-1]
            arr_inpaint_mask = arr_inpaint_mask[:, ::-1]
            arr_filter_mask = arr_filter_mask[:, ::-1]

        arr_gt = arr_gt.astype(np.float32) / 127.5 - 1
        arr_mask = arr_mask.astype(np.float32) / 255.0
        arr_mask2 = arr_mask2.astype(np.float32) / 255.0
        arr_mask3 = arr_mask3.astype(np.float32) / 255.0
        arr_inpaint_mask = arr_inpaint_mask.astype(np.float32) / 255.0
        arr_filter_mask = arr_filter_mask.astype(np.float32) / 255.0

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        if self.return_dict:
            name = os.path.basename(gt_path)
            return {
                'GT': np.transpose(arr_gt, [2, 0, 1]),
                'GT_name': name,
                'gt_keep_mask': np.transpose(arr_mask, [2, 0, 1]),
                'gt_keep_mask2': np.transpose(arr_mask2, [2, 0, 1]),
                'gt_keep_mask3': np.transpose(arr_mask3, [2, 0, 1]),
                'gt_inpaint_mask' : np.transpose(arr_inpaint_mask, [2, 0, 1]),
                'filter_mask': np.transpose(arr_filter_mask, [2, 0, 1])
            }
        else:
            raise NotImplementedError()

    def imread(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
