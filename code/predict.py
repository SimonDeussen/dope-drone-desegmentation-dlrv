from traceback import print_tb
from UNetSegmentation import utilities
from UNetSegmentation.model import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from evaluate import evalution_metics,prepare_plot


def make_predictions(model,imagePath,count):
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
		iou,dice=evalution_metics(gtMask,predMask)
		## Only plot the first 5 images
		if count<5:
			plot_title="IoU = "+np.array2string(np.round(iou),2)+"% Dice Score= "+np.array2string(np.round(dice,2))
			prepare_plot(orig, gtMask, predMask,plot_title)


	return iou,dice

        # load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(utilities.TEST_PATHS).read().strip().split("\n")
# imagePaths = np.random.choice(imagePaths,size=10)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(utilities.MODEL_PATH).to(utilities.DEVICE)

# iterate over the randomly selected test image paths
count=0
total_IoU=0
max_IoU=0
min_IoU=100
total_dice=0
max_dice=0
min_dice=100
for path in imagePaths:
	# make predictions and visualize the results
	IoU,dice=make_predictions(unet, path,count)
	## Saving max and min of dice and IoU
	if IoU>max_IoU:
		max_IoU=IoU

	if IoU<min_IoU:
		min_IoU=IoU

	if dice>max_dice:
		max_dice=dice

	if dice<min_dice:
		min_dice=dice

	total_IoU+=IoU   
	total_dice+=dice  





	count+=1  

average_iou=total_IoU/len(imagePaths)
average_dice=total_dice/len(imagePaths)
print(f"Average IoU is {average_iou} and max of {max_IoU} and min of {min_IoU}")
print(f"Average Dice is {average_dice} and max of {max_dice} and min of {min_dice}")



	
