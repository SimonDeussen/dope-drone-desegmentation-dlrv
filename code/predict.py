from traceback import print_tb
from UNetSegmentation import utilities
from UNetSegmentation.model import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from torchmetrics import IoU

def evaluate(groundTruth, prediction):

	#Adding a smoothing operation to avoid division 0 error
	EPS=1e-6
	origMask=groundTruth.copy()
	predMask=prediction.copy()


	intersection=(predMask&origMask).sum((0,1))
	union=(predMask | origMask).sum((0,1))
	iou=(intersection+EPS)/(union+EPS)

	print(f" The IOU is {iou}")

	return iou



def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	# set the layout of the figure and display it
	figure.tight_layout()
	# figure.show()
	plt.show()

def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	correct_train=0
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		# resize the image and make a copy of it for visualization
		# image = cv2.resize(image, (128, 128))
		orig = image.copy()
		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(utilities.MASK_DATASET_PATH,filename)
		#Change image path name to match the mask name
		groundTruthPath=groundTruthPath.replace("sat","mask")
		
		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		gtMask = cv2.imread(groundTruthPath,0)
		gtMask = cv2.resize(gtMask,dsize=(utilities.INPUT_IMAGE_HEIGHT,utilities.INPUT_IMAGE_WIDTH))
		# gtMask = cv2.resize(gtMask,dsize=(128,128))

        # make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(utilities.DEVICE)
		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		# filter out the weak predictions and convert them to integers
		predMask = (predMask > utilities.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)
		# correct_train += predMask.eq(gtMask.data).sum().item()

		# prepare a plot for visualization
		evaluate(gtMask,predMask)
		prepare_plot(orig, gtMask, predMask)

        # load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(utilities.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths,size=10)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(utilities.MODEL_PATH).to(utilities.DEVICE)
# test_model_path="/home/malika/Documents/Bonn_Stuff/DLRV/Project/dope-drone-desegmentation-dlrv/code/output/unet_tgs_forest_plot_100_epochs_64channel.pth"
# unet = torch.load(test_model_path).to(utilities.DEVICE)

# iterate over the randomly selected test image paths
count=0
for path in imagePaths:
	# make predictions and visualize the results
	make_predictions(unet, path)
	
