import os
import pickle
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import AutoProcessor

from .utils import parse_cloth_mask, parse_identity_image

CIHP_LABELS = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Glove": 3,
    "Sunglasses": 4,
    "UpperClothes": 5,
    "Dress": 6,
    "Coat": 7,
    "Socks": 8,
    "Pants": 9,
    "Torso-skin": 10,
    "Scarf": 11,
    "Skirt": 12,
    "Face": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Left-leg": 16,
    "Right-leg": 17,
    "Left-shoe": 18,
    "Right-shoe": 19,
}


class Transforms:
    def __init__(
        self, image_size: Tuple[int, int], use_augmentations: bool = True
    ) -> None:

        self.image_size = image_size
        self.use_augmentations = use_augmentations

        self.person_garment_size_transforms = A.Compose(
            [
                A.Resize(
                    *image_size, interpolation=cv2.INTER_LINEAR, always_apply=True
                ),
            ],
            additional_targets={
                "person_image": "image",
                "garment_image": "image",
                "densepose_image": "image",
            },
            is_check_shapes=False,
        )
        self.segmentation_size_transforms = A.Compose(
            [A.Resize(*image_size, interpolation=cv2.INTER_NEAREST, always_apply=True)]
        )

        if use_augmentations:
            self.all_spatial_transforms = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                ],
                additional_targets={
                    "garment_image": "image",
                    "person_image": "image",
                    "segmentation_image": "image",
                    "densepose_image": "image",
                },
            )
            self.person_spatial_transforms = A.Compose(
                [
                    A.ShiftScaleRotate(
                        rotate_limit=0,
                        shift_limit=0.2,
                        scale_limit=(-0.2, 0.2),
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.5,
                        value=0,
                    ),
                ],
                additional_targets={
                    "person_image": "image",
                    "segmentation_image": "image",
                    "densepose_image": "image",
                },
            )
            self.garment_spatial_transforms = A.Compose(
                [
                    A.ShiftScaleRotate(
                        rotate_limit=0,
                        shift_limit=0.2,
                        scale_limit=(-0.2, 0.2),
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.5,
                        value=0,
                    ),
                ],
                additional_targets={"garment_image": "image"},
            )
            self.person_garment_color_transforms = A.Compose(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.5
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.1, 0.02), contrast_limit=(-0.3, 0.3), p=0.5
                    ),
                ],
                additional_targets={"person_image": "image", "garment_image": "image"},
            )

        self.person_garment_type_transforms = A.Compose(
            [
                A.Normalize(mean=0.5, std=0.5, always_apply=True),
                ToTensorV2(),
            ],
            additional_targets={
                "person_image": "image",
                "garment_image": "image",
                "densepose_image": "image",
            },
        )

        def to_long(x, **kwargs):
            return x.to(torch.long)

        self.segmentation_type_transforms = A.Compose(
            [
                ToTensorV2(),
                A.Lambda(image=to_long, always_apply=True),
            ]
        )

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        # Due to albumentations requirement that we pass an `image` kwarg to the transforms
        # we create a dummy image key in the sample dictionary
        sample["image"] = sample["person_image"]

        sample = self.person_garment_size_transforms(**sample)
        sample["segmentation_image"] = self.segmentation_size_transforms(
            image=sample["segmentation_image"]
        )["image"]

        if self.use_augmentations:
            sample = self.all_spatial_transforms(**sample)
            sample = self.person_spatial_transforms(**sample)
            sample = self.garment_spatial_transforms(**sample)

            sample = self.person_garment_color_transforms(**sample)

        sample = self.person_garment_type_transforms(**sample)
        sample["segmentation_image"] = self.segmentation_type_transforms(
            image=sample["segmentation_image"]
        )["image"].to(torch.long)

        # Remove the dummy image key from the sample dictionary
        sample.pop("image")

        return sample


class SDCNVTONDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        pretrained_processor_path: str,
        garment_folder: str = "garment",
        person_folder: str = "person",
        segmentation_folder: str = "segmentation",
        densepose_folder: str = "densepose",
        pose_folder: str = "pose",
        background_label: Union[int, Tuple[int, ...]] = 0,
        identity_labels: Tuple[int, ...] = (1, 2, 4, 8, 9, 11, 12, 13, 16, 17, 18, 19),
        arm_labels: Tuple[int, ...] = (14, 15),
        cloth_labels: Tuple[int, ...] = (5, 6, 7),
        hand_mask_eps: float = 0.01,
        hand_mask_thickness: int = 10,
        transform: Callable[[Image.Image], torch.Tensor] = T.ToTensor(),
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.pretrained_processor_path = pretrained_processor_path
        self.garment_folder = garment_folder
        self.person_folder = person_folder
        self.segmentation_folder = segmentation_folder
        self.densepose_folder = densepose_folder
        self.pose_folder = pose_folder
        self.background_label = torch.tensor(background_label, dtype=torch.long)
        self.identity_labels = torch.tensor(identity_labels, dtype=torch.long)
        self.arm_labels = torch.tensor(arm_labels, dtype=torch.long)
        self.cloth_labels = torch.tensor(cloth_labels, dtype=torch.long)
        self.hand_mask_eps = hand_mask_eps
        self.hand_mask_thickness = hand_mask_thickness
        self.transform = transform

        self.processor = AutoProcessor.from_pretrained(pretrained_processor_path)
        self.folder_paths = [x for x in os.listdir(data_dir) if x.find("viton250k-gm")>=0]

    def __len__(self) -> int:
        return len(self.folder_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Select a random view of the garment from the garment folder
        garment_image_folder_path = os.path.join(
            self.data_dir, self.folder_paths[index], self.garment_folder
        )
        garment_image_filename = random.choice(os.listdir(garment_image_folder_path))

        # Select a random view of the person from the person folder
        person_image_folder_path = os.path.join(
            self.data_dir, self.folder_paths[index], self.person_folder
        )
        person_image_filename = random.choice(os.listdir(person_image_folder_path))

        # Select the corresponding segmentation image
        segmentation_image_folder_path = os.path.join(
            self.data_dir, self.folder_paths[index], self.segmentation_folder
        )
        segmentation_image_filename = f"{person_image_filename.split('.')[0]}.png"

        # Select the corresponding densepose image
        densepose_image_folder_path = os.path.join(
            self.data_dir, self.folder_paths[index], self.densepose_folder
        )
        densepose_image_filename = f"{person_image_filename.split('.')[0]}.png"

        # Select the corresponding pose
        pose_folder_path = os.path.join(
            self.data_dir, self.folder_paths[index], self.pose_folder
        )
        pose_filename = f"{person_image_filename.split('.')[0]}.pkl"

        # Load the images
        def load_image(folder, filename, fmt="RGB"):
            image = Image.open(os.path.join(folder, filename)).convert(fmt)
            return np.array(image)

        garment_image = load_image(garment_image_folder_path, garment_image_filename)
        person_image = load_image(person_image_folder_path, person_image_filename)
        segmentation_image = load_image(
            segmentation_image_folder_path, segmentation_image_filename, fmt="L"
        )
        densepose_image = load_image(
            densepose_image_folder_path, densepose_image_filename
        )

        cloth_mask = parse_cloth_mask(
            torch.Tensor(segmentation_image).unsqueeze(0), self.cloth_labels
        )
        # Load the pose
        with open(os.path.join(pose_folder_path, pose_filename), "rb") as f:
            pose = pickle.load(f)

        # Clip shares the same garment image but has a different preprocessing pipeline
        clip_garment_image = garment_image.copy()

        preprocessed_images = {
                "garment_image": garment_image,
                "person_image": person_image,
                "segmentation_image": segmentation_image,
                "densepose_image": densepose_image,
            }
        preprocessed_images = {x:self.transform(preprocessed_images[x]) for x in preprocessed_images}

        clip_image_encoder_garment_image = self.processor(
            images=clip_garment_image, return_tensors="pt"
        )["pixel_values"].squeeze(dim=0)

        identity_image, identity_mask = parse_identity_image(
            preprocessed_images["segmentation_image"],
            preprocessed_images["person_image"],
            pose,
            self.background_label,
            self.identity_labels,
            self.arm_labels,
            self.hand_mask_eps,
            self.hand_mask_thickness,
        )

        return {
            "clip_image_encoder_garment_image": clip_image_encoder_garment_image,
            "identity_image": identity_image,
            "identity_mask": identity_mask.float(),
            "cloth_mask": cloth_mask.float(),
            **preprocessed_images,
        }


@dataclass
class SDCNVTONDataModuleConfig:
    train_data_dir: str
    val_data_dir: str
    pretrained_processor_path: str = "openai/clip-vit-large-patch14"
    garment_folder: str = "garment"
    person_folder: str = "person"
    segmentation_folder: str = "segmentation"
    densepose_folder: str = "densepose"
    pose_folder: str = "pose"
    background_label: Union[int, Tuple[int, ...]] = 0
    identity_labels: Tuple[int, ...] = (1, 2, 4, 8, 11, 13, 18, 19)
    arm_labels: Tuple[int, ...] = (14, 15)
    cloth_labels: Tuple[int, ...] = (5, 6, 7)
    hand_mask_eps: float = 0.01
    hand_mask_thickness: int = 10
    image_size: Tuple[int, int] = (256, 256)
    train_batch_size: int = 4
    val_batch_size: int = 4
    train_num_workers: int = 4
    val_num_workers: int = 4
    train_shuffle: bool = True
    val_shuffle: bool = False
    pin_memory: bool = False
    persistent_workers: bool = False


class SDCNVTONDataModule(LightningDataModule):
    def __init__(self, config: SDCNVTONDataModuleConfig) -> None:
        super().__init__()

        self.config = config

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_transforms = Transforms(
            self.config.image_size, use_augmentations=True
        )
        self.train_dataset = SDCNVTONDataset(
            self.config.train_data_dir,
            self.config.pretrained_processor_path,
            self.config.garment_folder,
            self.config.person_folder,
            self.config.segmentation_folder,
            self.config.densepose_folder,
            self.config.pose_folder,
            self.config.background_label,
            self.config.identity_labels,
            self.config.arm_labels,
            self.config.cloth_labels,
            self.config.hand_mask_eps,
            self.config.hand_mask_thickness,
            transform=self.train_transforms,
        )

        self.val_transforms = Transforms(
            self.config.image_size, use_augmentations=False
        )
        self.val_dataset = SDCNVTONDataset(
            self.config.val_data_dir,
            self.config.pretrained_processor_path,
            self.config.garment_folder,
            self.config.person_folder,
            self.config.segmentation_folder,
            self.config.densepose_folder,
            self.config.pose_folder,
            self.config.background_label,
            self.config.identity_labels,
            self.config.arm_labels,
            self.config.cloth_labels,
            self.config.hand_mask_eps,
            self.config.hand_mask_thickness,
            transform=self.val_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            num_workers=self.config.train_num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            shuffle=self.config.train_shuffle,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.val_num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            shuffle=self.config.val_shuffle,
        )


if __name__ == "__main__":
    dm = SDCNVTONDataModule(
        SDCNVTONDataModuleConfig(
            train_data_dir="/data/data/merged_data_cihp",
            val_data_dir="/data/data/merged_data_cihp",
        )
    )
    dm.setup()

    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()

    for batch in train_dl:
        print("Train data keys", batch.keys())
        break

    for batch in val_dl:
        print("Val data keys", batch.keys())
        break