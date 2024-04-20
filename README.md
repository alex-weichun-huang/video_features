# Extracting Features from Videos

This directory contains the code to extract features from video datasets using mainstream vision models such as Slowfast, i3d, c3d, CLIP, etc. The only requirement for you is to provide a list of videos that you would like to extract features from in your input directory. 

If you have any question with the code, please contact whuang288@wisc.edu for further support.

<br>

# Structure of the Repository

- ` models/`: This is the directory where you customize the models (e.g. structure, preprocessing, cropping) you want to use for feature extraction.

- ` configs/`: This directory contains configuration files that define different runtime settings

- ` condor/`: This directory contains the code to submit computation to CHTC for throughput computing, ignore if not applicable.

<br>


# Installation

* Follow [INSTALL.md](INSTALL.md) for installing necessary dependencies.

<br>

# Instructions

* Pass in the path to the config file. 

* Make sure to modify the fields in the config file if necessary

```sh
python main.py --config configs/egovlp.yaml 
```

> **Note:** For egovlp, i3d, and c3d, you are REQUIRED to manually download pretrained <a href = "https://drive.google.com/drive/folders/1Qe_9XLUJELB69gYwSpAo1DU3yZ64BOpg"> checkpoints </a>. Please refer to [models/README](models/README.md) for more details.


<br>

##  Customization

- To custoimze preprocessing (transformation, cropping, and mirroring), please modify the implementation of the `get_transform` method under models/`model_name`.py

- To use customized model or your own checkpoints, please modify the implementation of the `load_model` method under models/`model_name.py`


<br>

## Adding a new Model

1. Add a new yaml file to `configs/`. Check out [configs/README](configs/README.md) for more instructions.

2. Add a new python file to `models/`. Check out [models/README](models/README.md) for more instructions.

<br>

# Acknowledgement

This directory is built on top of the <a href = "https://github.com/facebookresearch/Ego4d"> Ego4d directory </a>.

<br>

# Appendix

To resize the video or to convert the video into a different frame rate, refer to this <a href = "https://github.com/whuang288alex/resize_videos"> repository </a>.
