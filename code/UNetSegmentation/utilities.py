from typing import Tuple
import torch
import os
import random

##Path to data folder
CURRENT_PATH=os.getcwd()
DATASET_PATH=os.path.join(CURRENT_PATH,"data")
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH,"images")
MASK_DATASET_PATH=os.path.join(DATASET_PATH,"masks")
print(f"the data path is at {DATASET_PATH} \n and the masks are at {MASK_DATASET_PATH}\n")
##Train | Validation | Test Split
SPLIT=[0.5,0,25,0.25]
#Test data use 15% of all data rest for training 
TEST_SPLIT=0.15
##Check if GPU available 
if torch.cuda.is_available():
    DEVICE="cuda"
    PIN_MEMORY=True
else:
    DEVICE="cpu"
    PIN_MEMORY=False        
print(f"Available device is {DEVICE} of name {torch.cuda.get_device_name(torch.cuda.current_device())}\n")

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
## not used right now but can be used to choose how many levels
# NUM_CHANNELS = 3
# NUM_CLASSES = 1
# NUM_LEVELS = 3 
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 32 ## My  PC stop working at 128 
# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_model_forests.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])



    






