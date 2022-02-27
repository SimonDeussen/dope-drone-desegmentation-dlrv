<!-- Link to [Meistertask](https://www.meistertask.com/app/project/rbtDaOGT/h-brs-dope-dlvr) -->
This is the repository for the deep learning for semenatic segmentation project. There are two methods available.
The data set used for this project can be found at https://www.kaggle.com/balraj98/deepglobe-land-cover-classification-dataset

1. Pure U-Net  
2. Ensemble of U-Net with VGG19, ResNet and Inception16 backbone,

<h3>To run the pure U-Net</h3>

1. Add dataset of your project files in a folder and add that path to "code/assets/config.py"
    - You will have to define the original images and ground truth masks folders paths separately
2. Change any other parameters you need in that config file
3. To train the network run `python code/train.py`
4. To predict run `python code/predict.py`


<h3>To run the pure ensembled U-Net</h3>

1. Add dataset of your project files in a folder
2. Go to "notebooks/U-net with backbones.ipynb" 
    - First change the path to ground truth and masks
3. Then run the file using jupyter-notebook or similar software
