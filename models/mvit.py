# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from typing import Tuple, Any

from models.common import FeedVideoInput, ThreeCrop, Mirror
from pytorchvideo.models.hub.vision_transformers import mvit_base_16, mvit_base_32x3
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Identity, Module
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo


def load_model(
    inference: Any,
    config: Any,
    patch_final_layer: bool = True,
) -> Module:
    
    if config.use_remote:
        assert config.ckpt in ("k400")
        if config.ckpt == "k400":
            print("Loading remote K400 MViT")
            model = mvit_base_32x3(pretrained=True)
    else:
        raise ValueError("Local MViT ckpt not available")
        
    assert model is not None

    if patch_final_layer:
        model.head = Identity()

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
        
    # image-based dataset
    if config.ckpt == "imagenet":
        # NOTE untested due to MViT imagenet not not available on torch hub
        transforms += [Lambda(lambda x: x.squeeze_(2))]
    return Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(transforms),
            )
        ]
    )