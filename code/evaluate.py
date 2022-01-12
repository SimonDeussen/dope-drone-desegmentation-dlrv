from traceback import print_tb
from typing import IO
from matplotlib import figure
import numpy as np
import cv2
import os
from imutils import paths
from UNetSegmentation import utilities
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




def evalution_metics(groundTruth,prediction):
    """
    This implementation is only for a two class segmentation
  
    """
    

	#Adding a smoothing operation to avoid division 0 error
    EPS=1e-6
    origMask=groundTruth.copy()
    print(f"Orig shape {origMask.shape}")
    print(origMask)
    origMask=np.where(origMask==255,1,origMask)
    predMask=prediction.copy()
    predMask=np.where(predMask==255,1,predMask)
    intersection=(predMask&origMask).sum((0,1))    
    union=(predMask | origMask).sum((0,1))
    iou=((intersection+EPS)/(union+EPS))*100
    
    dice=2.*(predMask*origMask).sum() / (predMask+origMask).sum()
    # print(f" The IOU is {iou}")
    # print(f"The F1 score is {dice}")
    
    return iou,dice


def prepare_plot(origImage, origMask, predMask,metric):
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
    figure.suptitle(metric)
    # figure.show()
    plt.show()


if __name__=="__main__":

    CURRENT_PATH=os.getcwd()
    DATASET_PATH=os.path.join(CURRENT_PATH,"data/zain_results")
    IMAGE_PATH=os.path.join(DATASET_PATH,"original_images")
    PREDICTIONS_PATH=os.path.join(DATASET_PATH,"predictions")
    GROUNDTRUTH_PATH=os.path.join(DATASET_PATH,"labels")



    imagePaths = sorted(list(paths.list_images(IMAGE_PATH)))
    groundtruthPaths=sorted(list(paths.list_images(GROUNDTRUTH_PATH)))
    maskPaths = sorted(list(paths.list_images(PREDICTIONS_PATH)))


    for index in range(len(imagePaths)):
        oriImage=mpimg.imread(imagePaths[index])
        groundTruthImage=mpimg.imread(groundtruthPaths[index])
        maskImage=mpimg.imread(maskPaths[index])
        IoU,dice=evalution_metics(groundTruthImage,maskImage)
        plot_title="IoU = "+np.array2string(np.round(IoU),2)+"% Dice Score= "+np.array2string(np.round(dice,2))
        prepare_plot(oriImage,groundTruthImage,maskImage,plot_title)
       
        if index==3:
            break

