# Import global packages and libraries that are required 
# by the methods included in the file.

import numpy as np
import pandas as pd
import os
import random
from skimage import io
from skimage.transform import resize
import datetime
import glob
import random as rn
import gc

import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2

from scipy import stats
from statistics import mean 
from statistics import median 
import seaborn as sns

from concurrent import futures
import threading
from collections import Counter

from numpy import expand_dims

from IPython.display import Markdown
from enum import Enum


class SKIN_CANCER_TUMOR_TYPE(Enum):
    
    '''
    An ENUM class to contain the enumeration values for Red Blood Cell categories (Parasitized or Uninfected) 
    in this study.
    
    Tumors can be benign (noncancerous) or malignant (cancerous). 
    '''
    
    BENIGN = 'benign'
    MALIGNANT = 'malignant'


class dataUtils():
    
    '''
    A Utility/Helper class that contains helper methods for some exploratry image based analysis.
    The methods majorly includes extracting the statistical level details of the images and plotting 
    various pre-processing and augmentation level transformations on sample images. 
    '''
    
    def __init__(self):
        pass
    
    
    @staticmethod
    def PrintMarkdownText(textToDisplay):
        
        '''
        Purpose: 
            A static method to display markdown formatted output like bold, italic bold etc..

        Parameters:
            1. textToDisplay - the string message with formatting styles that is to be displayed

        Return Value: 
            NONE
        '''
        
        display(Markdown('<br>'))
        display(Markdown(textToDisplay))

    
    def GetLabelledSkinCancerData(self):
        
        '''
        Purpose: 
            Creates a dataframe with the filenames of all the skin cancer images and the 
            corresponding labels. The dataframe has 2 columns - 'filename' and 'label'

        Parameters: 
            NONE

        Return Value: 
            The computed skin cancer dataframe.
        '''
        
        benign_images = glob.glob('ISIC Skin Cancer/images/benign/*.jpg')
        malignant_images = glob.glob('ISIC Skin Cancer/images/malignant/*.jpg')
        len(benign_images), len(malignant_images)

        skin_cancer_df = pd.DataFrame({
            'filename': benign_images + malignant_images,
            'label': ['Benign'] * len(benign_images) + ['Malignant'] * len(malignant_images)
        })

        # Shuffle the rows in the dataset
        skin_cancer_df = skin_cancer_df.sample(frac=1, random_state=34).reset_index(drop=True)

        return skin_cancer_df

   
    def GetImageDirectory(self, imageCategory):
        
        '''
        Purpose: 
            Gets the directory path for the category of image (benign/malignant) in the dataset.

        Parameters:
            1. imgCategory - The category of the image (benign/malignant). See SKIN_CANCER_TUMOR_TYPE ENUM.

        Return Value: 
            The image directory path.
        '''

        # Get the correct directory path based on the category of image
        if imageCategory == SKIN_CANCER_TUMOR_TYPE.BENIGN.value:
            imgDirectory = 'ISIC Skin Cancer/images/benign/*.jpg'
        else:
            imgDirectory = 'ISIC Skin Cancer/images/malignant/*.jpg'

        return imgDirectory


    
    def GetAllImageShape(self, imageCategory):
        
        '''
        Purpose: 
            Extract the Image dimensions of the images in the dataset for the given imageCategory 
            (benign/malignant). As the number of images are large, this method utilizes parallel 
            processing using the ThreadPoolExecutor for faster computation.

        Parameters:
            1. imgCategory - The category of the image (benign/malignant). See SKIN_CANCER_TUMOR_TYPE ENUM.

        Return Value: 
            A list of dinemsions of all the images of the concerned imageCategory.
        '''
        
        images = []
        
        '''A nested function to get the image shape that is called parallely from the ThreadPoolExecutor
        Parameter:
            img - The image that is to be resized.'''
        def GetImageShape(img):
            return cv2.imread(img).shape

        # Get the correct directory path based on the category of image
        imgDirectory = self.GetImageDirectory(imageCategory)

        for img_path in glob.glob(imgDirectory):
            images.append(img_path)
        
        # https://docs.python.org/3/library/concurrent.futures.html for details on max_workers
        executer = futures.ThreadPoolExecutor(max_workers=None)
        all_image_dimension_map = executer.map(GetImageShape, [img for img in images])

        return Counter(all_image_dimension_map)

    
    def ReadAllImages(self, imageList, resizeImage = False, newImageSize = None):
        
        '''
        Purpose: 
            Resizes all the images passed in to the new dimension defined in 'newImageSize'. 
            As the number of images are large, this method utilizes parallel processing using the 
            ThreadPoolExecutor for faster computation.

        Parameters:
            1. imageList - The list of all the images that are to be re-sized.
            2. newImageSize - The size to which the images have to be re-sized.

        Return Value: 
            List of the resized images.
        '''
        
        '''A nested function to resize the image to the specified new dimension 
        and is called parallely from the ThreadPoolExecutor.
        Parameter:
            img - The image that is to be resized.'''
        def ResizeImage(img):
            
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # if resize is set to True, then resize the image to the new dimension.
            if resizeImage:
                img = cv2.resize(img, dsize = newImageSize, interpolation = cv2.INTER_CUBIC)
                
            img = np.array(img, dtype=np.float32)
            return img
        
        # https://docs.python.org/3/library/concurrent.futures.html for details on max_workers
        executer = futures.ThreadPoolExecutor(max_workers=None)
        img_data_map = executer.map(ResizeImage, [image for image in imageList])

        return np.array(list(img_data_map))
    
    
    def ReadAndDisplayInputImages(self, imageCategory, numImagesToDisplay):
        
        '''
        Purpose: 
            Read and display the first 5 images of the imageCategory (benign/malignant) 
            in the dataset.

        Parameters:
            1. imgCategory - The category of the image (benign/malignant). See SKIN_CANCER_TUMOR_TYPE ENUM.
            2. numImagesToDisplay - Total number of images to display.

        Return Value: 
            NONE
        '''
        
        images = []
        
        # Get the correct directory path based on the category of image
        imgDirectory = self.GetImageDirectory(imageCategory)

        # Read the first 5 images
        for img_path in glob.glob(imgDirectory):
            if len(images) < numImagesToDisplay:
                images.append(img.imread(img_path))

        # Display the images
        plt.figure(figsize=(20,10))
        columns = 5
        for i, image in enumerate(images):
            plt.subplot(len(images) / columns + 1, columns, i + 1)
            plt.imshow(image)
            plt.axis('off')

    
            
    def DisplayAnnotatedImages(self, df, numImagesToDisplay):
        
        '''
        Purpose: 
            Display the given number of annotated images from the passed in dataframe of image 
            filenames and labels.

        Parameters:
            1. df - The dataset which contails the filenames and the corresponding labels.
            2. numImagesToDisplay - Total number of images to display.

        Return Value: 
            NONE
        '''
        
        images = []
        labels = []

        # Get the correct directory path based on the category of image
        # imgDirectory = self.GetImageDirectory(imageCategory)

        # Read the first 'numImagesToDisplay' images.
        for img_path in df.filename:
            if len(images) < numImagesToDisplay:
                # extract the corresponding image label
                label = df.loc[df['filename'] == img_path].label
                labels.append(label)
                images.append(img.imread(img_path))

        # Display the images
        plt.figure(figsize=(20,10))
        columns = 5
        for i, image in enumerate(images):
            plt.subplot(len(images) / columns + 1, columns, i + 1)
            plt.imshow(image)
            plt.title(labels[i].item())
            plt.axis('off')
    
    
  
    def ComputeAndPlotImageDimensionalStatistics(self, imgCategory):
        
        '''
        Purpose: 
            Computed and displays the dimensional statistics of the images in the directory and
            plots the distribution for the X and Y dimensional component.

        Parameters:
            1. imgCategory - The category of the image (benign/malignant). See SKIN_CANCER_TUMOR_TYPE ENUM.

        Return Value: 
            NONE
        '''
        
        dim_x = []
        dim_y = []
        
        allImageDims = self.GetAllImageShape(imgCategory)
        
        setOfUniqueDimensions = set(allImageDims)
        
        for shape in allImageDims:
            x,y,channel = shape
            dim_x.append(x)
            dim_y.append(y)


        f, ax = plt.subplots(1, 2)
        f.set_figwidth(10)

        sns.distplot(dim_x, kde=True, fit=stats.gamma, ax=ax[0]);
        sns.distplot(dim_y, kde=True, fit=stats.gamma, ax=ax[1]);

        ax[0].title.set_text('Distribution of X Dimension')
        ax[1].title.set_text('Distribution of Y Dimension')

        plt.show()
        
        print('Statistical Features - Image Dimension:')
        print('---------------------------------------')
        print('Max X Dimension:', max(dim_x))
        print('Min X Dimension:', min(dim_x))
        print('Mean X Dimension:', mean(dim_x))
        print('Median X Dimension:', median(dim_x))
        print('---------------------------------------')
        print('Max Y Dimension:', max(dim_y))
        print('Min Y Dimension:', min(dim_y))
        print('Mean Y Dimension:', mean(dim_y))
        print('Median Y Dimension:', median(dim_y))
        
        print('\nTotal # Images with Unique Dimensions:', len(setOfUniqueDimensions))

    
    
    def GetSampleImage(self, imgCategory):
        
        '''
        Purpose:
            Reads and returns the first image of the given category - (benign/malignant).

        Parameters:
            1. imgCategory - The category of the image (benign/malignant). See SKIN_CANCER_TUMOR_TYPE ENUM.

        Return Value: 
            NONE
        '''
        
        images = []

        # Get the correct directory path based on the category of image
        imgDirectory = self.GetImageDirectory(imgCategory)

        # Read and return the first image in the directory.
        for img_path in glob.glob(imgDirectory):
            if len(images) < 1:
                images.append(img.imread(img_path))

        return images[0]
    