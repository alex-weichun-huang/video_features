# Requirements

To set up the environment with conda:

```sh
conda create --name vid_feats python=3.9
conda activate vid_feats
```

Please visit the PyTorch official website and follow the <a href="https://pytorch.org/get-started/locally/">installation guide</a> to install torch, torchvision, and torchaudio that works for your system. The following command work for CUDA 11.7:

```sh
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

Install other dependencies:

```sh
python -m pip install -r requirements.txt
```