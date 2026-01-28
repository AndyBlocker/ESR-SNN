# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100
from util.neuromorphic_datasets import Cifar10DVS
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from spikingjelly.datasets import dvs128_gesture

from glob import glob
from io import BytesIO

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from PIL import Image

class TransformFirstDataset(Dataset):
    """
    Apply a transform to the first element of a dataset sample.

    This is useful for datasets that return (data, target, *extras), where we want
    data augmentation to run inside DataLoader workers (instead of the training loop).
    """

    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.transform is None:
            return sample

        if isinstance(sample, tuple) and len(sample) > 0:
            x = self.transform(sample[0])
            return (x,) + sample[1:]
        if isinstance(sample, list) and len(sample) > 0:
            sample = list(sample)
            sample[0] = self.transform(sample[0])
            return sample
        return sample

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset == "imagenet":
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    elif args.dataset == "cifar100":
        dataset = CIFAR100(root=args.data_path, train=True if is_train else False, download=True, transform=transform)
    elif args.dataset == "cifar10":
        dataset = CIFAR10(root=args.data_path, train=True if is_train else False, download=True, transform=transform)
    elif args.dataset == "cifar10dvs":
        if is_train:
            dataset = Cifar10DVS(root=args.data_path, resolution=(args.input_size, args.input_size))[0]
        else:
            dataset = Cifar10DVS(root=args.data_path, resolution=(args.input_size, args.input_size))[1]
    elif args.dataset == "dvs128":
        if is_train:
            dataset = dvs128_gesture.DVS128Gesture(root=args.data_path, train=True, data_type='frame', frames_number=16,
                                                    split_by='number')
        else:
            dataset = dvs128_gesture.DVS128Gesture(root=args.data_path, train=False, data_type='frame', frames_number=16,
                                                    split_by='number')
    else:
        raise NotImplementedError

    print(dataset)

    return dataset



# def _resolve_parquet_splits(data_path: str):
#     train_files = sorted(glob(os.path.join(data_path, "train-*.parquet")))
#     val_files   = sorted(glob(os.path.join(data_path, "validation-*.parquet")))
#     test_files  = sorted(glob(os.path.join(data_path, "test-*.parquet")))

#     if len(train_files) == 0:
#         raise FileNotFoundError(f"No train-*.parquet found in: {data_path}")
#     if len(val_files) == 0 and len(test_files) == 0:
#         raise FileNotFoundError(f"No validation-*.parquet or test-*.parquet found in: {data_path}")

#     data_files = {"train": train_files}
#     if len(val_files) > 0:
#         data_files["validation"] = val_files
#     if len(test_files) > 0:
#         data_files["test"] = test_files
#     return data_files


# def _pick_keys(ds):
#     cols = list(ds.column_names)

#     # image key
#     if "image" in cols:
#         image_key = "image"
#     else:
#         cand = [c for c in cols if "image" in c.lower() or "img" in c.lower()]
#         if not cand:
#             raise KeyError(f"Cannot find image column in parquet dataset. columns={cols}")
#         image_key = cand[0]

#     # label key
#     if "label" in cols:
#         label_key = "label"
#     elif "labels" in cols:
#         label_key = "labels"
#     else:
#         cand = [c for c in cols if "label" in c.lower() or "class" in c.lower()]
#         cand = [c for c in cand if c != image_key]
#         if not cand:
#             raise KeyError(f"Cannot find label column in parquet dataset. columns={cols}")
#         label_key = cand[0]

#     return image_key, label_key


# def _to_pil(x):
#     # 1) already PIL
#     if isinstance(x, Image.Image):
#         return x.convert("RGB")

#     # 2) HF Image-like dict: {"bytes":..., "path":...} or {"path":...}
#     if isinstance(x, dict):
#         if x.get("bytes", None) is not None:
#             return Image.open(BytesIO(x["bytes"])).convert("RGB")
#         if x.get("path", None) is not None:
#             return Image.open(x["path"]).convert("RGB")

#     # 3) raw bytes
#     if isinstance(x, (bytes, bytearray, memoryview)):
#         return Image.open(BytesIO(bytes(x))).convert("RGB")

#     # 4) numpy array (H,W,C) or (H,W)
#     if isinstance(x, np.ndarray):
#         if x.ndim == 2:
#             return Image.fromarray(x).convert("RGB")
#         if x.ndim == 3:
#             return Image.fromarray(x.astype(np.uint8)).convert("RGB")

#     # 5) torch tensor -> PIL
#     if torch.is_tensor(x):
#         arr = x.detach().cpu().numpy()
#         if arr.ndim == 2:
#             return Image.fromarray(arr).convert("RGB")
#         if arr.ndim == 3:
#             # (C,H,W) -> (H,W,C) if needed
#             if arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[-1]:
#                 arr = np.transpose(arr, (1, 2, 0))
#             return Image.fromarray(arr.astype(np.uint8)).convert("RGB")

#     raise TypeError(f"Unsupported image type: {type(x)}")


# class HFParquetImageNetTorch(Dataset):
#     def __init__(self, hf_ds, image_key: str, label_key: str, transform=None, label_map=None):
#         self.ds = hf_ds
#         self.image_key = image_key
#         self.label_key = label_key
#         self.transform = transform
#         self.label_map = label_map  # 可选：str->int

#     def __len__(self):
#         return len(self.ds)

#     def __getitem__(self, idx):
#         ex = self.ds[idx]
#         img = _to_pil(ex[self.image_key])

#         y = ex[self.label_key]
#         # label 可能是 numpy/int/str
#         if isinstance(y, (np.integer,)):
#             y = int(y)
#         elif isinstance(y, str):
#             if self.label_map is None:
#                 raise ValueError("Label is str but label_map is None.")
#             y = int(self.label_map[y])
#         else:
#             y = int(y)

#         if self.transform is not None:
#             img = self.transform(img)

#         return img, y


# def build_dataset(is_train, args):
#     transform = build_transform(is_train, args)

#     if args.dataset != "imagenet":
#         raise NotImplementedError(f"Unknown dataset: {args.dataset}")

#     # 1) 找到所有 parquet shard
#     data_files = _resolve_parquet_splits(args.data_path)

#     # 2) 用 HF load_dataset 直接读 parquet
#     #    cache_dir / num_proc 都是可选参数：你没定义也不影响
#     ds_dict = load_dataset(
#         "parquet",
#         data_files=data_files,
#         cache_dir=getattr(args, "hf_cache_dir", None),
#         num_proc=getattr(args, "hf_num_proc", None),
#     )

#     # 3) 选 split：训练用 train；验证优先 validation，没有就退化到 test
#     if is_train:
#         split_name = "train"
#     else:
#         split_name = "validation" if "validation" in ds_dict else "test"

#     hf_ds = ds_dict[split_name]
#     image_key, label_key = _pick_keys(hf_ds)

#     # 4) 如果 label 是字符串，建立 train split 的映射（保证一致）
#     label_map = None
#     try:
#         sample = hf_ds[0]
#         y0 = sample[label_key]
#         if isinstance(y0, str):
#             train_ds = ds_dict["train"]
#             # 取 train 的 unique label（一次性扫一遍）
#             uniq = sorted(set(train_ds[label_key]))
#             label_map = {name: i for i, name in enumerate(uniq)}
#     except Exception:
#         pass

#     dataset = HFParquetImageNetTorch(
#         hf_ds,
#         image_key=image_key,
#         label_key=label_key,
#         transform=transform,
#         label_map=label_map,
#     )

#     print(f"[HF parquet] split={split_name} size={len(dataset)} image_key={image_key} label_key={label_key}")
#     return dataset


# def build_dataset_mae(transform, args):

#     # if args.dataset != "imagenet":
#     #     raise NotImplementedError(f"Unknown dataset: {args.dataset}")

#     # 1) 找到所有 parquet shard
#     data_files = _resolve_parquet_splits(args.data_path)

#     # 2) 用 HF load_dataset 直接读 parquet
#     #    cache_dir / num_proc 都是可选参数：你没定义也不影响
#     ds_dict = load_dataset(
#         "parquet",
#         data_files=data_files,
#         cache_dir=getattr(args, "hf_cache_dir", None),
#         num_proc=getattr(args, "hf_num_proc", None),
#     )

#     # 3) 选 split：训练用 train；验证优先 validation，没有就退化到 test
#     split_name = "train"

#     hf_ds = ds_dict[split_name]
#     image_key, label_key = _pick_keys(hf_ds)

#     # 4) 如果 label 是字符串，建立 train split 的映射（保证一致）
#     label_map = None
#     try:
#         sample = hf_ds[0]
#         y0 = sample[label_key]
#         if isinstance(y0, str):
#             train_ds = ds_dict["train"]
#             # 取 train 的 unique label（一次性扫一遍）
#             uniq = sorted(set(train_ds[label_key]))
#             label_map = {name: i for i, name in enumerate(uniq)}
#     except Exception:
#         pass

#     dataset = HFParquetImageNetTorch(
#         hf_ds,
#         image_key=image_key,
#         label_key=label_key,
#         transform=transform,
#         label_map=label_map,
#     )

#     print(f"[HF parquet] split={split_name} size={len(dataset)} image_key={image_key} label_key={label_key}")
#     return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN if not args.define_params else args.mean
    std = IMAGENET_DEFAULT_STD if not args.define_params else args.std
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            no_aug=args.no_aug,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    crop_pct = 0.9
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
