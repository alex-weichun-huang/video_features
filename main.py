import warnings
warnings.filterwarnings("ignore")

import os
import yaml
import argparse
from typing import List
from src.load import get_videos
from src.config import Video, load_config
from src.extraction import perform_feature_extraction


def print_stats_for_videos(
    all_videos: List[Video], videos: List[Video]
):  
    # stats for uids
    assert isinstance(all_videos[0], Video)
    assert isinstance(videos[0], Video)

    print(
        f"""
    Total Number of Videos = {len(all_videos)}
    Incomplete videos = {len(videos)}
    """
    )


def run_feature_extraction(config):
    
    assert os.path.exists(config.io.video_dir_path),  "The video path provided in the config file does not exist."
    os.makedirs(config.io.out_path, exist_ok=True)

    
    # load all the videos in the specified directory and filter out the ones that have already been processed
    videos, all_videos = get_videos(config)
    assert len(videos) > 0, "Please check if (1) the input directory is empty or (2) ALL the videos have already been processed (feature files already exist in the output directory)."
    print_stats_for_videos(all_videos=all_videos, videos=videos)
    
    
    # save the config file
    print("###################### Feature Extraction Config ####################")
    print(yaml.dump(config.toDict()))
    print("#####################################################################")
    with open(f"{config.io.out_path}/config.yaml", "w") as out_f:
        out_f.write(yaml.dump(config.toDict()))
        
    # perform feature extraction
    perform_feature_extraction(videos, config)
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='path to the config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_feature_extraction(cfg)
