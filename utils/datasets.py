import random

import numpy as np
import torch
from torch import Tensor
import torchvision.transforms.functional as TF
from torchvision import transforms

from utils.data_utils import pil_loader

from .data_constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
    IMAGE_TASKS
)
from .dataset_folder import (
    MultiTaskPatientBalancedDatasetFolder
)


class DataAugmentationForMultiMAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        self.rgb_mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        self.rgb_std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.input_size = args.input_size
        self.hflip = args.hflip
        self.intensity_shift = args.intensity_shift
        if len(args.random_crop) > 0:
            parts = args.random_crop.split("--")
            self.random_crop = [float(p) for p in parts]
        else:
            self.random_crop = []
        self.affine = transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.8, 1.1),
            shear=0,
            fill=0
        )
        self.input_size = args.input_size
        self.args = args

    def crop(self, image: Tensor, crop_params: tuple) -> Tensor:
        """Crop the image to the specified size starting from (start_h, start_w)."""
        start_h, start_w, size = crop_params
        return image[..., start_h:start_h + size, start_w:start_w + size]

    def get_crop_params(self, image: Tensor, rel_size_low=0.8, rel_size_high=1.0):
        """Get random crop parameters for the image."""
        h, w = image.shape
        crop_size = random.randint(int(h * rel_size_low), int(h * rel_size_high))
        start_h = random.randint(0, h - crop_size)
        start_w = random.randint(0, w - crop_size)
        return start_h, start_w, crop_size

    def __call__(self, task_dict):
        flip = random.random() < self.hflip  # Stores whether to flip all images or not
        affine = self.affine
        affine_params = affine.get_params(
            affine.degrees, affine.translate, affine.scale, affine.shear,
            [512, 512]
        )
        crop_params = None
        if len(self.random_crop) > 0:
            crop_params = self.get_crop_params(
                next(iter(task_dict.values())),
                rel_size_low=self.random_crop[0],
                rel_size_high=self.random_crop[1]
            )
        if flip and 'laterality' in task_dict:
            task_dict['laterality'] = 1 - task_dict['laterality']
        for task in set(task_dict.keys()) & set(IMAGE_TASKS):
            if flip:
                task_dict[task] = np.flip(task_dict[task], axis=-1)
            if self.intensity_shift > 0 and task not in ['layermaps', 'bscanlayermap']:
                # IMPORTANT: It assumes that the image is in [0, 1] range
                shift = np.random.normal(0, self.intensity_shift, 1).astype(np.float32)
                if random.random() < 0.5:
                    shift = -shift
                task_dict[task] = np.clip(task_dict[task] + shift, 0, 1)

            img = torch.from_numpy(task_dict[task].copy()).contiguous().unsqueeze(0)

            if task in ['bscan', 'bscanlayermap']:
                # All transformations
                c_params = affine_params
            else:
                # Only translation in x direction
                c_params = 0, (affine_params[1][0],0), affine_params[2], 0
            if task in ['layermaps', 'bscanlayermap']:
                interpolation = TF.InterpolationMode.NEAREST
            else:
                interpolation = TF.InterpolationMode.BILINEAR
            if not self.args.no_affine:
                img = TF.affine(  # type: ignore
                    img,
                    *c_params,
                    interpolation=interpolation,
                    fill=0,  # type: ignore
                    center=affine.center,
                )
            if crop_params is not None:
                img = self.crop(img, crop_params)

            if img.shape[1:] != tuple(self.input_size[task]):
                img = TF.resize(
                    img,
                    self.input_size[task],
                    interpolation=interpolation,
                )

            if task in ['layermaps', 'bscanlayermap']:
                img = img.squeeze(0)
            task_dict[task] = img

        if 'bscanlayermap' in task_dict.keys():
            mask = task_dict['bscanlayermap']
            # Define your range of lesion classes
            lesion_range = torch.arange(12, 19, device=mask.device)
            # Use broadcasting to check all classes at once
            # mask[..., None] adds a new dimension to compare against the range
            # .any() checks if that class exists anywhere in the mask
            presence = (mask[..., None] == lesion_range).any(dim=0).any(dim=0)
            task_dict['lesion_presence'] = presence.float()

        return task_dict

    def __repr__(self):
        repr = "(DataAugmentationForMultiMAE,\n"
        repr += ")"
        return repr


def build_patient_balanced_multimae_pretraining_dataset(args):
    transform = DataAugmentationForMultiMAE(args)
    print(f'Data from {args.data_path}')
    return MultiTaskPatientBalancedDatasetFolder(
        args.data_path,
        args.all_domains,
        args=args,
        transform=transform,
        loader=pil_loader,
    )
