# Modifying/Adding config files

> **Note:** In this context, a video is segmented into multiple "clips," and a feature vector is extracted from each "clip."

- `video_dir_path`: The directory (absolute path) where the input videos are stored. All videos in this directory will be processed.

- `out_path`: The directory (absolute path) where the extracted features will be stored.

- `batch_size`: The number of "clips" that get fed into the model at each iteration.

- `frame_window`: The number of frames in a "clip."

- `stride`: The number of frames between the first frame of two neighboring "clips."

- `use_remote`: Whether you want to use a remote checkpoint or a local one for the model weights.

- `ckpt`: The name/path of the model checkpoint (can be either local or remote, depending on `use_remote`).

- `side_size`: The size to which the shorter side of the input video will be resized.

- `crop`: You can choose between "three_crops" and "center." This option determines how the input frames will
 be cropped. For more customization, please go to `models/model_name.py` and modify the `get_transform` method.

- `crop_size`: The size of the crop mentioned above.

- `model_module_str`: The location of the Python file where you implement the model (default: under `models/`).