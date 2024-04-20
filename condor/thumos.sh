#!/bin/bash
# TODO: add your own job commands here. Please refer to https://chtc.cs.wisc.edu/uw-research-computing/htc/guides.html
# Below is the common workflow:

# transfer files (inputs, code, and packed environment) from the staging directory
tar -xzf /staging/groups/li_group_biostats/datasets/THUMOS14_test.tar.gz
tar -xzf /staging/groups/li_group_biostats/datasets/THUMOS14_val.tar.gz
tar -xzf /staging/groups/li_group_biostats/video_feature_extraction/video_feature_extraction.tar.gz
mkdir feature_extraction
tar -xzf /staging/groups/li_group_biostats/video_feature_extraction/feature_extraction.tar.gz -C feature_extraction

# set up the environment
set -e
export PATH
. feature_extraction/bin/activate

# prepare input
mkdir videos
mv test/*.mp4 videos/
mv validation/*.mp4 videos/
cp /staging/groups/li_group_biostats/datasets/thumos_video_list.txt videos/

# run the extraction
python main.py --config configs/egovlp.yaml