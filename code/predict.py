
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

from assets.config import *

def prepare_plot(origImage, origMask, predMask):
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)

    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")

    figure.tight_layout()
    figure.show()

# TODO refactor this method to only return the preMask!
def make_prediction(model, image_path):
    model.eval()

    with torch.no_grad():
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(device)
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        pred_mask = model(image).squeeze()
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.cpu().numpy()
        # filter out the weak predictions and convert them to integers
        pred_mask = (pred_mask > THRESHOLD) * 255
        pred_mask = pred_mask.astype(np.uint8)
        return pred_mask


def visualize_prediction(model, image_path):
    pred_mask = make_prediction(model, image_path)

    filename = image_path.split(os.path.sep)[-1]
    mask_path = os.path.join(MASK_PATH, filename).replace("sat", "mask")

    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_HEIGHT))

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0

    prepare_plot(image, mask, pred_mask)



print("[INFO] loading up test image paths...")
image_paths = open(TEST_PATHS).read().strip().split("\n")
image_paths = np.random.choice(image_paths, size=10)

print("[INFO] load up model...")
unet = torch.load(MODEL_PATH).to(device)

for path in image_paths:
	visualize_prediction(unet, path)