# we need the os modules for working with the paths
from os.path import join, isdir
import torch

NUM_CHANNELS = 3
NUM_CLASSES = 1
NUM_LEVELS = 3


INIT_LR = 0.001
NUM_EPOCHS = 4
BATCH_SIZE = 128
TEST_SPLIT = 0.15


INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256

THRESHOLD = 0.5
BASE_OUTPUT = "output"

MODEL_PATH = join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = join(BASE_OUTPUT, "plot.png")
TEST_PATHS = join(BASE_OUTPUT, "test_paths.txt")

# for windows vs linux reasons, otherwise it wont work '/' vs '\'
# just define the data folder for your system and the rest works

DATA_FOLDER = "C:\\Users\\Simon\\Documents\\code\\_data\\Forest Segmented\\"
IMAGE_PATH = join(DATA_FOLDER, 'Forest Segmented', 'images')

MASK_PATH = join(DATA_FOLDER, 'Forest Segmented', 'masks')


assert isdir(IMAGE_PATH), f"Image folder does not exist - check you path in assets/config.py: {IMAGE_PATH}"
assert isdir(MASK_PATH), f"Mask folder does not exist - check you path in assets/config.py: {MASK_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.get_device_name(torch.cuda.current_device()))

PIN_MEMORY = True if device == "cuda" else False
