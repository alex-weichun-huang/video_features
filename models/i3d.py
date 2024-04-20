# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
from typing import Tuple, Any

import torch
from models.common import FeedVideoInput, ThreeCrop, Mirror
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Identity, Module
from torchvision.transforms import Compose, Lambda, TenCrop
from torchvision.transforms._transforms_video import  CenterCropVideo, NormalizeVideo

from .i3d_arch import build_i3d


def load_model(
    inference: Any,
    config: Any,
    patch_final_layer: bool = True,
) -> Module:
    
    assert config.use_remote is False, "Remote model not supported"
    assert config.ckpt is not None, "Checkpoint not provided"
    
    print("Loading I3D")    # takes RGB / XY input
    model = build_i3d(pretrained=config.ckpt)

    assert model is not None
    if patch_final_layer:
        model.logits = Identity()

    # Set to GPU or CPU
    model = FeedVideoInput(model)
    model = model.eval()
    model = model.to(inference.device)
    return model


def get_transform(inference: Any, config: Any):
    transforms = [
        Lambda(lambda x: x[:, ::config.dilation]),
        Lambda(lambda x: x / 255.0),
        NormalizeVideo(config.mean, config.std),
        ShortSideScale(size=config.side_size),
    ]
    
    if config.crop == "center":
        transforms.append(CenterCropVideo(config.crop_size))
    elif config.crop == "three_crops":
        transforms.append(ThreeCrop(config.crop_size))
    elif config.crop == "ten_crops":
        transforms.append(TenCrop(config.crop_size))
        transforms.append(Lambda(lambda crops: torch.stack(crops)))
    if config.mirror:
        transforms.append(Mirror())
        
    return Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(transforms),
            )
        ]
    )
