from typing import Iterator, Tuple

import cv2
import numpy as np
import torch
from controlnet_aux.open_pose import HandResult, PoseResult
from einops import rearrange
from torch import nn


def draw_handpose(
    canvas: np.ndarray, keypoints: HandResult, eps: float = 0.01, thickness: float = 30
) -> np.ndarray:
    if not keypoints:
        return canvas

    H, W, C = canvas.shape

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for e1, e2 in edges:
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        if k1 is None or k2 is None:
            continue

        x1 = int(k1.x * W)
        y1 = int(k1.y * H)
        x2 = int(k2.x * W)
        y2 = int(k2.y * H)
        if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), thickness=thickness)

    return canvas


def generate_hand_mask(
    poses: list[PoseResult],
    segmentation_image: torch.FloatTensor,
    arm_labels: torch.LongTensor,
    eps: float = 0.01,
    thickness: float = 10,
) -> torch.FloatTensor:
    # Generate numpy canvas same size as person image
    C, H, W = segmentation_image.shape
    mask = np.zeros((H, W, C), dtype=np.uint8)
    for pose in poses:
        mask = draw_handpose(mask, pose.left_hand, eps=eps, thickness=thickness)
        mask = draw_handpose(mask, pose.right_hand, eps=eps, thickness=thickness)

    # Convert to torch tensor
    pose_hand_mask = torch.from_numpy((mask > 0)).to(torch.bool)
    pose_hand_mask = rearrange(pose_hand_mask, "h w c -> c h w")

    # Obtain the arm mask from the segmentation image
    segmentation_hand_mask = torch.isin(segmentation_image, arm_labels)

    # Combine the pose hand mask and the segmentation hand mask
    hand_mask = pose_hand_mask & segmentation_hand_mask

    return hand_mask


def transform_to_rectangular_mask(mask: torch.BoolTensor) -> torch.BoolTensor:
    # Return original mask if there are no True values
    if not mask.any():
        return mask

    # If mask is 3D, then perform or operation along the first dimension
    combined_mask = mask.any(dim=0) if mask.ndim == 3 else mask

    # Find the bounds of the mask
    rows = combined_mask.any(dim=1)
    cols = combined_mask.any(dim=0)
    top, bottom = torch.where(rows)[0][[0, -1]]
    left, right = torch.where(cols)[0][[0, -1]]

    # Create a new mask with the bounds
    new_mask = torch.full_like(combined_mask, False)
    new_mask[top : bottom + 1, left : right + 1] = True

    # Replicate the mask along the first dimension if it was 3D
    new_mask = new_mask.expand_as(mask) if mask.ndim == 3 else new_mask

    return new_mask


def parse_identity_image(
    segmentation_image: torch.LongTensor,
    person_image: torch.FloatTensor,
    pose: list[PoseResult],
    background_label: torch.LongTensor,
    identity_labels: torch.LongTensor,
    arm_labels: torch.LongTensor,
    eps: float = 0.01,
    thickness: float = 10,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    # Obtain the identity mask
    identity_mask = torch.isin(segmentation_image, identity_labels)
    hand_mask = generate_hand_mask(pose, segmentation_image, arm_labels, eps, thickness)
    identity_mask = identity_mask | hand_mask

    # Obtain the non-background mask (1s where the segmentation image is not the background)
    non_background_mask = ~torch.isin(segmentation_image, background_label)
    non_background_mask = transform_to_rectangular_mask(non_background_mask)

    # Select the background pixels from the person image
    background_image = torch.where(
        non_background_mask, torch.zeros_like(person_image), person_image
    )

    # Select the identity pixels from the person image
    identity_image = torch.where(identity_mask, person_image, background_image)

    # Combine the identity mask and the non-background mask
    identity_mask = identity_mask | ~non_background_mask

    # Select only the first channel
    identity_mask = identity_mask[:1, ...]

    return identity_image, identity_mask


def batch_parse_identity_image(
    segmentation_images: torch.LongTensor,
    person_images: torch.FloatTensor,
    poses: list[list[PoseResult]],
    background_label: torch.LongTensor,
    identity_labels: torch.LongTensor,
    arm_labels: torch.LongTensor,
    eps: float = 0.01,
    thickness: float = 10,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    identity_images, identity_masks = [], []
    for segmentation_image, person_image, pose in zip(
        segmentation_images, person_images, poses
    ):
        identity_image, identity_mask = parse_identity_image(
            segmentation_image,
            person_image,
            pose,
            background_label,
            identity_labels,
            arm_labels,
            eps,
            thickness,
        )
        identity_images.append(identity_image)
        identity_masks.append(identity_mask)

    return torch.stack(identity_images, dim=0), torch.stack(identity_masks, dim=0)


def parse_cloth_mask(
    segmentation_image: torch.LongTensor, cloth_labels: torch.Tensor
) -> torch.FloatTensor:
    # Obtain the cloth masks
    cloth_mask = torch.isin(segmentation_image, cloth_labels)

    # Select only the first channel
    cloth_mask = cloth_mask[:1, ...]

    return cloth_mask


def batch_parse_cloth_mask(
    segmentation_images: torch.LongTensor, cloth_labels: torch.Tensor
) -> torch.FloatTensor:
    # Obtain the cloth masks
    cloth_masks = torch.isin(segmentation_images.to(torch.long), cloth_labels)

    # Select only the first channel
    cloth_masks = cloth_masks[:, :1, ...]

    return cloth_masks


def pad_dims_like(x: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    ndim = other.ndim - x.ndim
    return x.view(*x.shape, *((1,) * ndim))


def _update_ema_weights(
    ema_weight_iter: Iterator[torch.Tensor],
    online_weight_iter: Iterator[torch.Tensor],
    ema_decay_rate: float,
) -> None:
    for ema_weight, online_weight in zip(ema_weight_iter, online_weight_iter):
        if ema_weight.data is None:
            ema_weight.data.copy_(online_weight.data)
        else:
            ema_weight.data.lerp_(online_weight.data, ema_decay_rate)


def update_ema_model_(
    ema_model: nn.Module, online_model: nn.Module, ema_decay_rate: float
) -> nn.Module:
    # Update parameters
    _update_ema_weights(
        ema_model.parameters(), online_model.parameters(), ema_decay_rate
    )
    # Update buffers
    _update_ema_weights(ema_model.buffers(), online_model.buffers(), ema_decay_rate)

    return ema_model


def zero_init_(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        nn.init.zeros_(param)
    return module
