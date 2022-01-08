import torch
from torch.utils.data import Dataset
import cv2


class DataSet(Dataset):    

    def __init__(self,
                imagePaths:list,
                maskPaths: list,
                transforms=None):
        '''
        The init data structurre
        '''


        self.imagePaths=imagePaths
        self.maskPaths=maskPaths
        self.transforms=transforms
        # self.inputs_dtype = torch.float32
        # self.targets_dtype = torch.long


    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, 
                    index:int):
        '''
        Get the files from 
        '''

        imagePath=self.imagePaths[index]
        image=cv2.imread(imagePath)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask=cv2.imread(self.maskPaths[index],0)
        if self.transforms is not None:
            image=self.transforms(image)
            mask=self.transforms(mask)
            

        return (image,mask)


    




    
