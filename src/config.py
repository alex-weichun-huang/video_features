import yaml
import importlib
from dotmap import DotMap
from torchaudio.transforms import Resample
from torchvision.transforms import Compose, Lambda
from dataclasses import dataclass


@dataclass
class Video:
    uid: str
    path: str
    w: int
    h: int
    frame_count: int
    frame_rate: int = 30
    has_audio: bool = False
    is_stereo: bool = False
    
    @property
    def dim(self) -> int:
        return (self.w * self.h) / (2 if self.is_stereo else 1)
    
    
DEFAULTS ={
    'io': {
        'filter_completed': True,
        'debug_mode': False,
        'video_dir_path': '/inputs/videos_ego4d',
        'out_path': '/features/c3d',
    },
    'inference': {
        'device': 'cuda',
        'batch_size': 1,
        'num_workers': 8,
        'prefetch_factor': 2,
        'frame_window': 32,
        'stride': 16,
    },
    'model': {
        'use_remote': True,
        'ckpt': None,
        'mirror': False,
        'crop': None,
        'crop_size': 112,
        'side_size': 288,
        'dilation': 1,
        'mean': (0.45, 0.45, 0.45),
        'std': (0.225, 0.225, 0.225),
        'slowfast_alpha': 4,
    },
    'model_module_str': ''
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_file):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(DEFAULTS, config)
    config = DotMap(config)
    return config


def get_model_module(config):
    """
    helper function to get model specific functions
    """
    return importlib.import_module(config.model_module_str)


def load_model(config, patch_final_layer=True):
    """
    helper function to get model specific model
    """
    module = get_model_module(config)
    return module.load_model(
        config.inference,
        config.model,
        patch_final_layer=patch_final_layer,
    )


def get_transform(config):
    """
    helper function to get model specific transform
    """
    ic = config.inference
    nc = ic.norm_config
    model_transform = get_model_module(config).get_transform(ic, config.model)
    
    transforms = []
    
    if hasattr(config, "norm_config") and config.norm_config.normalize_audio:
        print(f"Normalizing with: {config.norm_config}")
        def resample_audio(x):
            return Resample(
                orig_freq=x["audio_sample_rate"],
                new_freq=nc.resample_audio_rate,
                resampling_method=nc.resampling_method,
            )
        transforms += [Lambda(resample_audio)]
        
    transforms += [model_transform]
    return Compose(transforms)