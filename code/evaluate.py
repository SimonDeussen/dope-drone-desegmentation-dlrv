from traceback import print_tb
import numpy as np
import cv2
import os
from imutils import paths
from UNetSegmentation import utilities


def evalution_metics(groundTruth,prediction):
    """
    This implementation is only for a two class segmentation
  
    """
    

	#Adding a smoothing operation to avoid division 0 error
    EPS=1e-6
    origMask=groundTruth.copy()
    origMask=np.where(origMask==255,1,origMask)
    predMask=prediction.copy()
    predMask=np.where(predMask==255,1,predMask)
    intersection=(predMask&origMask).sum((0,1))    
    union=(predMask | origMask).sum((0,1))
    iou=(intersection+EPS)/(union+EPS)
    total_pixels=(2*origMask.shape[0]*origMask.shape[1])
    dice=2.*(predMask*origMask).sum() / (predMask+origMask).sum()
    print(f" The IOU is {iou}")
    print(f"The F1 score is {dice}")
    
    return iou,dice



if __name__=="__main__":

    imagePaths = sorted(list(paths.list_images(utilities.IMAGE_DATASET_PATH)))
