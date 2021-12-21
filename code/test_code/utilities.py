from typing import Tuple
import torch
import os
import random

def main():
    print("Starting utilities")

    ##Path to 
    DATASET_PATH=os.path.join("data")
    MASK_PATH=os.path.join("masks")

    ##Train | Validation | Test Split

    print(f"the data path is at {DATASET_PATH} \n and the masks are at {MASK_PATH}\n")

    SPLIT=[0.5,0,25,0.25]

    if torch.cuda.is_available():
        DEVICE="cuda"
        PIN_MEMORY=True



    else:
        DEVICE="cpu"
        PIN_MEMORY=False
        
    print(f"Available device is {DEVICE}")








if __name__=="__main":

    main()


    # print("Starting utilities")

    # ##Path to 
    # DATASET_PATH=os.path.join("data")
    # MASK_PATH=os.path.join("masks")

    # ##Train | Validation | Test Split

    # print(f"the data path is at {DATASET_PATH} \n and the masks are at {MASK_PATH}\n")

    # SPLIT=[0.5,0,25,0.25]

    # if torch.cuda.is_available():
    #     DEVICE="cuda"
    #     PIN_MEMORY=True



    # else:
    #     DEVICE="cpu"
    #     PIN_MEMORY=False
        
    # print(f"Available device is {DEVICE}")



