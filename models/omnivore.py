# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
from typing import Tuple, Any

import torch
from models.common import ThreeCrop, Mirror
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Identity, Module
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo


class WrapModel(Module):
    def __init__(self, model: Module, input_type: str):
        super().__init__()
        self.model = model
        self.input_type = input_type

    def forward(self, x) -> torch.Tensor:
        input = x['video']
        if input.ndim == 5:
            out = self.model(input, input_type=self.input_type)
        elif input.ndim == 6:
            # evaluate one crop at a time
            out = [self.model(input[:, i], input_type=self.input_type) \
                    for i in range(input.shape[1])]
            out = torch.stack(out, dim=-1).mean(dim=-1)
        else:
            raise ValueError('invalid input size')
        return out


def load_model(
    inference: Any,
    config: Any,
    patch_final_layer: bool = True,
) -> Module:
    
    if config.use_remote:
        print("Loading remote Omnivore")
        model = torch.hub.load("facebookresearch/omnivore", model=config.ckpt)
    else:
        assert config.ckpt is not None, "Checkpoint not provided"
        print("Loading local Omnivore")
        model = torch.load(config.ckpt)
        
    if patch_final_layer:
        model.heads.image = Identity()
        model.heads.video = Identity()
        model.heads.rgbd = Identity()

    # Set to GPU or CPU
    model = WrapModel(model, "video")
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
    
    if config.mirror:
        transforms.append(Mirror())
    
    return ApplyTransformToKey(
        key="video",
        transform=Compose(transforms),
    )
