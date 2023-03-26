from traceback import print_tb
from typing import IO
from matplotlib import figure
import numpy as np
import cv2
import os
from imutils import paths
from UNetSegmentation import utilities
import matplotlib.pyplot as plt

def pixel_Accuracy(pred, label):
    pixel_labeled = np.sum(label > 0)
    #print(pixel_labeled)
    pixel_correct = np.sum((pred == label) * (label > 0))
    pixel_accuracy = 1.0 * pixel_correct / (pixel_labeled + 1e-10)
    
    return pixel_accuracy, pixel_correct, pixel_labeled


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
    print(CURRENT_PATH)
    DATASET_PATH=os.path.join(CURRENT_PATH,"data/zain_results")
    IMAGE_PATH=os.path.join(DATASET_PATH,"original_images")
    PREDICTIONS_PATH=os.path.join(DATASET_PATH,"predictions")
    GROUNDTRUTH_PATH=os.path.join(DATASET_PATH,"labels")



    imagePaths = sorted(list(paths.list_images(IMAGE_PATH)))
    groundtruthPaths=sorted(list(paths.list_images(GROUNDTRUTH_PATH)))
    predictionPaths = sorted(list(paths.list_images(PREDICTIONS_PATH)))

    total_IoU=0
    max_IoU=0
    min_IoU=100
    pixel_accuracy=[]
    pixel_correct=[]
    pixel_labeled=[]
    
    for index in range(0,len(imagePaths)):
        oriImage=cv2.imread(imagePaths[index])
        groundTruthImage=cv2.imread(groundtruthPaths[index],0)
        predictionImage=cv2.imread(predictionPaths[index],0)
        IoU,dice=evalution_metics(groundTruthImage,predictionImage)
        pix_acc, pix_corr, pix_labeled = pixel_Accuracy(predictionImage, groundTruthImage)
        pixel_accuracy.append(pix_acc)
        pixel_correct.append(pix_corr)
        pixel_labeled.append(pix_labeled) 


        if IoU>max_IoU:
            max_IoU=IoU

        if IoU<min_IoU:
            min_IoU=IoU
        total_IoU+=IoU       

        if index<5:
            #Prnints only the first 4 plots

            plot_title="IoU = "+np.array2string(np.round(IoU),2)+"% Dice Score= "+np.array2string(np.round(dice,2))
            prepare_plot(oriImage,groundTruthImage,predictionImage,plot_title)

    pixel_accuracy = np.array(pixel_accuracy)
    pixel_correct = np.array(pixel_correct)
    pixel_labeled = np.array(pixel_labeled)
    acc = 100.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    
    
    average_iou=total_IoU/len(imagePaths)
    print(f"Average IoU is {average_iou} and max of {max_IoU} and min of {min_IoU}")
    
    print(f"Mean pixel accuracy: {acc} %")







if __name__=="__main__":

    ##For original Unet use dataset =1
    ##For ensemble dataset use dataset=2
    main()
    