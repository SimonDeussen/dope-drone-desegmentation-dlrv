from typing import Tuple
import torch
import os
import random

##Path to data
CURRENT_PATH=os.getcwd()
DATASET_PATH=os.path.join(CURRENT_PATH,"data")
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH,"images")
MASK_DATASET_PATH=os.path.join(DATASET_PATH,"masks")
##Train | Validation | Test Split
print(f"the data path is at {DATASET_PATH} \n and the masks are at {MASK_DATASET_PATH}\n")
SPLIT=[0.5,0,25,0.25]
TEST_SPLIT=0.15
if torch.cuda.is_available():
    DEVICE="cuda"
    PIN_MEMORY=True
else:
    DEVICE="cpu"
    PIN_MEMORY=False        
print(f"Available device is {DEVICE}")

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 64
# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_model_forests.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])



    






