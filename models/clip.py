# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import clip
import torch
from typing import Tuple, Any

from torch.nn import Module
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale

from models.common import ThreeCrop, Mirror
from .clip_arch import load


class WrapModel(Module):
    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def forward(self, x) -> torch.Tensor:
        input = x['video']
        if input.ndim == 5:
            print("input.shape: ", input.shape)
            out = self.model(input[:,:,0])
        elif input.ndim == 6:
            # input.shape: [1(bs), 6(mirror + 3crop), 3(color), 8(T), 224(w), 224(h)]
            # output.shape: [1(bs), 512(feature), 6(mirror + 3crop)] -> [1, 512, 2(mirror), 3(3crop)]
            # because CLIP is an image model, we only throw in the first frame of each clip
            out = torch.stack([self.model(input[:, i, :, 0]) for i in range(input.shape[1])], dim = -1).view(input.shape[0], -1, 2)
        else:
            raise ValueError('invalid input size')
        return out


def load_model(
    inference: Any,
    config: Any,
    patch_final_layer: bool = True,
) -> Module:
    
    assert config.ckpt is not None
    
    if config.use_remote:
        print("Loading remote clip...")
        path = str(config.ckpt)
        
        # change the path name to the remote naming convention
        if path == "ViT-B-16":
            path = "ViT-B/16"
        elif path == "ViT-B-32":
            path = "ViT-B/32"
        elif path == "ViT-L-14":
            path = "ViT-L/14"
        elif path == "ViT-L-14-336px":
            path = "ViT-L/14@336px"
            
        model, preprocess = clip.load(path, device = inference.device)
    else:
        print("Loading local clip...")
        model = load(config.ckpt, device = inference.device)
    
    model = model.encode_image
    model = WrapModel(model)
    model.eval()
    return model
   
    
def get_transform(inference: Any, config: Any):
    transforms = [
        Lambda(lambda x: x[:, ::config.dilation]),
        Lambda(lambda x: x[torch.tensor([2, 1, 0]), :, :, :]),
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
    
    return ApplyTransformToKey(
        key="video",
        transform=Compose(transforms),
    )
