from traceback import print_tb
from typing import IO
from matplotlib import figure
import numpy as np
import cv2
import os
from imutils import paths
from UNetSegmentation import utilities
import matplotlib.pyplot as plt




def evalution_metics(groundTruth,prediction):
    """
    This implementation is only for a two class segmentation
  
    """
    

	#Adding a smoothing operation to avoid division 0 error
    EPS=1e-6
    origMask=groundTruth.copy()  
    predMask=prediction.copy()
    intersection=(predMask&origMask).sum((0,1))    
    union=(predMask | origMask).sum((0,1))
    iou=((intersection+EPS)/(union+EPS))*100    
    dice=2.*(predMask*origMask).sum() / (predMask+origMask).sum()
    
    
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


def main():

    CURRENT_PATH=os.getcwd()

    DATASET_PATH=os.path.join(CURRENT_PATH,"data/zain_results")
    IMAGE_PATH=os.path.join(DATASET_PATH,"original_images")
    PREDICTIONS_PATH=os.path.join(DATASET_PATH,"predictions")
    GROUNDTRUTH_PATH=os.path.join(DATASET_PATH,"labels")



    imagePaths = sorted(list(paths.list_images(IMAGE_PATH)))
    groundtruthPaths=sorted(list(paths.list_images(GROUNDTRUTH_PATH)))
    maskPaths = sorted(list(paths.list_images(PREDICTIONS_PATH)))

    total_IoU=0
    max_IoU=0
    min_IoU=100
    total_dice=0
    max_dice=0
    min_dice=100

    for index in range(0,len(imagePaths)):
        oriImage=cv2.imread(imagePaths[index])
        groundTruthImage=cv2.imread(groundtruthPaths[index],0)
        maskImage=cv2.imread(maskPaths[index],0)
        IoU,dice=evalution_metics(groundTruthImage,maskImage)

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
        if index<5:

            #Prnints only the first 4 plots

            plot_title="IoU = "+np.array2string(np.round(IoU),2)+"% Dice Score= "+np.array2string(np.round(dice,2))
            prepare_plot(oriImage,groundTruthImage,maskImage,plot_title)


    average_iou=total_IoU/len(imagePaths)
    average_dice=total_dice/len(imagePaths)

    print(f"Average IoU is {average_iou} and max of {max_IoU} and min of {min_IoU}")
    print(f"Average Dice is {average_dice} and max of {max_dice} and min of {min_dice}")








if __name__=="__main__":

    ##The main file run the zain net 
    main()
    