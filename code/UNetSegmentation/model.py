from torch._C import _set_mkldnn_enabled
from . import utilities
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch


class Block(Module):

    def __init__(self,
                inChannels,
                outChannels) -> None:
        super().__init__()
        self.conv1=Conv2d(inChannels,outChannels,3)
        self.relu=ReLU()
        self.conv2=Conv2d(inChannels,outChannels,3)


    def forward(self,x):
        '''
        Applies the two convolutions and return it
        
        '''
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(Module):
    def __init__(self,channels=[3,16,32,64]) -> None:
        super().__init__()
        #store the encoder block and maxpool layer 
        self.encBlocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels)-1)]
        )
        self.pool =MaxPool2d(2)


    def forward(self,x):
        ##stores the intermediate outputs
        blockOutputs=[]

        for block in self.encBlocks:
            x=block(x)
            blockOutputs.append(x)
            x=self.pool(x)

        #Return intermdiate outputs
        return blockOutputs


class Decoder(Module):
    def __init__(self,channels=[64,32,16]) -> None:
        super().__init__()
        self.channels = channels
        ##The decoder part that used transpose convolution
        self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
        ##The decoder blocks
        self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])



    def forward(self,x,encFeatures):

        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
		# return the final decoder output
        return x

    def crop(self,encFeatures,x):
        ##Use the dimensions of the input image
        ## and does a center crop to mathch the dimension
        (_, _, H, W) = x.shape
        encFeatures=CenterCrop([H,W])(encFeatures)


        return encFeatures

class UNet(Module):
    '''
    The UNet architecture based on the sumodules 
    '''
    def __init__(self,encChannels=(3, 16, 32, 64),
		 decChannels=(64, 32, 16),
		 nbClasses=1, retainDim=True,
		 outSize=(utilities.INPUT_IMAGE_HEIGHT,  utilities.INPUT_IMAGE_WIDTH)) -> None:
         super().__init__()
         self.encoder=Encoder(encChannels)
         self.decoder=Decoder(encChannels)

         #Initialize the regression head store the class variables 
         ## check the kernal size 
         self.head = Conv2d(decChannels[-1],nbClasses,1)
         self.retainDim= retainDim
         self.outSize=outSize

    def forward(self, x):
		# grab the features from the encoder
        encFeatures = self.encoder(x)
		# pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0],
			encFeatures[::-1][1:])
		# pass the decoder features through the regression head to
		# obtain the segmentation mask
        map = self.head(decFeatures)
		# check to see if we are retaining the original output
		# dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = F.interpolate(map, self.outSize)
		# return the segmentation map
        return map



		


