import os
import sys
import torch
import torch.nn as nn
from .video_transformer import SpaceTimeTransformer


sys.path.append(os.path.dirname(__file__))
import parse_config # we need to import this to load the checkpoint that Meta Research provided


class EgoVLPv2(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = SpaceTimeTransformer(num_frames=4)
        
        # load the checkpoint that Meta Research provided; only load the video_model part
        ckpt = torch.load(os.path.join(os.path.dirname(__file__), 'assets/egovlpv2.pth'), map_location='cpu')
        self.model.load_state_dict(ckpt)
        
        # freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        
        x1 = self.model(x) # 768
        x2 = self.model.norm(x1) # 768
        x3 = self.model.pre_logits(x2) # 768
        feat = torch.cat([x1, x2, x3], dim=1) # 2304
        return feat


def build_egovlpv2():
    return EgoVLPv2()