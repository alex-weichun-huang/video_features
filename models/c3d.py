# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from typing import Tuple, Any

import torch
from models.common import FeedVideoInput, ThreeCrop, Mirror
from torchvision.transforms import (
    Compose, Resize, CenterCrop, FiveCrop, TenCrop, Lambda
) 
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Identity, Module
from .c3d_arch import build_c3d, load_mean


def load_model(
    inference: Any,
    config: Any,
    patch_final_layer: bool = True,
) -> Module:
    
    assert config.use_remote is False, "Remote model not supported"
    assert config.ckpt is not None, "Checkpoint not provided"
    
    print("Loading C3D")    # takes RGB / XY input
    model = build_c3d(config.ckpt)
    assert model is not None
    
    # Set to GPU or CPU
    model = FeedVideoInput(model)
    model = model.eval()
    model = model.to(inference.device)
    return model


def get_transform(inference: Any, config: Any):
    
    mean = load_mean(config.ckpt)
    assert mean is not None
    mean = mean.cpu()
    
    transforms = [
        Lambda(lambda x: x[:, ::config.dilation]),
        Lambda(lambda x: x / 255.0),
        # NormalizeVideo(config.mean, config.std),
        ShortSideScale(size=config.side_size),
    ]
    
    # handle crops
    if config.crop == 'center':
        transforms.append(CenterCrop(config.crop_size))
        transforms.append(Lambda(lambda crops: crops - mean))
        
    elif config.crop == 'five_crops':
        transforms.append(FiveCrop(config.crop_size))
        transforms.append(Lambda(lambda crops: torch.stack(crops)))
        transforms.append(Lambda(lambda crops: crops - mean))
        
    elif config.crop == 'ten_crops':
        transforms.append(TenCrop(config.crop_size))
        transforms.append(Lambda(lambda crops: torch.stack(crops)))
        transforms.append(Lambda(lambda crops: crops- mean))
    else:
        raise NotImplementedError()
    
    # handle mirror
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
