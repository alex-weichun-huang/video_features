# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from typing import Tuple, Any

from models.common import FeedVideoInput, ThreeCrop, Mirror
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Module
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from .egovlpv2_arch import build_egovlpv2


def load_model(
    inference: Any,
    config: Any,
    patch_final_layer: bool = True,
) -> Module:
    
    # NOTE: EgoVLP only has 1 checkpoint so we don't need to specify it
    assert config.use_remote is False, "Remote model not supported"
    
    print("Loading EgoVLP")
    model = build_egovlpv2()

    assert model is not None

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
    else :
        raise ValueError('Need to either use center crop or three crop')

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
