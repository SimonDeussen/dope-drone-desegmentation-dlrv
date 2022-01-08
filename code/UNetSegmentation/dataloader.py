import torch
from torch.utils import data
import cv2



class DataSet(data.Dataset):    

    def __init__(self,
                imagePaths:list,
                maskPaths: list,
                transforms=None):
        '''
        The init data structurre
        '''


        self.input_paths=imagePaths
        self.mask_paths=maskPaths
        self.transforms=transforms
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long


    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, 
                    index:int):
        '''
        Get the files from 
        '''

        input_path=self.input_paths[index]
        image=cv2.imread(input_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask=cv2.imread(self.mask_paths[index],0)

        if self.transforms is not None:
            image=self.transforms(image)
            mask=self.transforms(mask)


                    


        return image,mask


    




    
