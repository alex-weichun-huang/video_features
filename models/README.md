# Adding a model

- Create a python file with model name as file name, and include the following methods
  
  - get_transform
   
  - load_model
  
  - addition functions for your model

- Useful functions in common.py

  - FeedVideoInput
  
  - FeedVideoInputList
  
  - Mirror
  
  - ThreeCrop
 
# Using Local Checkpoints

If you want to use the local checkpoints (either pretrained or fine-tuned)

1. Download the <a href = "https://drive.google.com/drive/folders/1Qe_9XLUJELB69gYwSpAo1DU3yZ64BOpg"> checkpoints </a>and add them to `./models/model_name_arch/assets`

2. Set the variable `use_remote` in config files to false and set the variable `ckpt` in the config file to your desired checkpoints name/path.
