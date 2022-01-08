from imutils import paths
from UNetSegmentation import utilities
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
if __name__=="__main__":
    print("Start of test")
    imagePaths = sorted(list(paths.list_images(utilities.IMAGE_DATASET_PATH)))
    maskPaths = sorted(list(paths.list_images(utilities.MASK_DATASET_PATH)))


    print(len(imagePaths))

    image1=mpimg.imread(imagePaths[30])
    print(image1.shape)
    plt.imshow(image1)
    plt.show()
    # print(imagePaths[0],maskPaths[0])

    
    # img1=cv2.imread(imagePaths[0])  
    # win_name='image'
    # print(img1.shape)
    # cv2.imshow('image',img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    


