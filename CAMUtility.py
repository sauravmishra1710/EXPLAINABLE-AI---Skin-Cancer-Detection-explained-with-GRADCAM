'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
DESCRIPTION: CAM class activation visualization.
             A Utility/Helper class that contains helper methods for visualizing the class level activations of 
             a Convolotional Neural Network. CAM visualizations help us know which regions, patterns of 
             the image the neural network is looking at, and activating the region around those patterns. To 
             visualize the activation maps, we would need the output of the LAST CONVOLUTIONAL LAYERS
             and the final CLASSIFICATION LAYERS.
             
REFERENCE:   https://arxiv.org/abs/1512.04150

WEBSITE:     https://arxiv.org/abs/1512.04150
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Import Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

import cv2
import numpy as np

# Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class CAMUtilities():

    '''
    A Utility/Helper class that contains helper methods for visualizing the class level activations of 
    a Convolotional Neural Network. CAM visualizations help us know which regions, patterns of 
    the image the neural network is looking at, and activating the region around those patterns. To 
    visualize the activation maps, we would need the output of the LAST CONVOLUTIONAL LAYERS
    and the final CLASSIFICATION LAYERS.
    
    Args:
        model (keras model): the classification model.
        
        layerName (string): name of the layer for which the
                   class activations are to be visualized.
    
    '''
    
    def __init__(self, model, layerName):
        
        self.model = model
        self.layerName = layerName

        last_conv_layer = self.model.get_layer(self.layerName).output
        self.cammodel = Model(inputs=self.model.input,
                                       outputs=[last_conv_layer, self.model.output])
    
    def GetImageArrayInBatch(self, img_path, size):
        
        '''
        Purpose: 
            Function to extract the image array.

        Parameters:
            1. img_path - the path of the image for which the array is required.
            2. size - the preferred size of the image to be read.

        Return Value: 
            img_array - The array representation (in terms of the batch - 
                        (1, size_x, size_y, channel)) of the image.
        
        '''

        img = keras.preprocessing.image.load_img(img_path, target_size=size)

        # Get the image array
        img_array = keras.preprocessing.image.img_to_array(img)

        # We add a dimension to transform our array into a "batch"
        # of size (1, size_x, size_y, channel)
        img_array = np.expand_dims(array, axis=0)
        
        return img_array

    def compute_heatmap(self, image):
        
        '''
        Purpose: 
            Get the class level activations heatmap for the CNN network. The activation maps help us 
            identify & localize the regions, patterns of the image the neural network looks at, and 
            activates the region around the patterns. To visualize the activation maps, this function 
            works on the output of the LAST CONVOLUTIONAL LAYER and the final CLASSIFICATION LAYERS.

        Parameters:
            1. img_array - the array representation of the image for which the activation maps are to be 
                           visualized.
            2. model - the CNN model whose activations are to be analyzed.
            3. last_conv_layer_name - name of the last convolutional layer of the model.
            4. classifier_layer_names - the final classification layers.

        Return Value: 
            heatmap - The heatmap showing the more active regions the CNN looked at in deciding the class
                      for the image.
                      
        '''
        
        # conv_outputs is the output activations of the layer 
        # pointed to by the self.layerName
        # predictions is the predicted result of the image.
        [conv_outputs, predictions] = self.cammodel.predict(image)
        # print(conv_outputs.shape) --> (1, 33, 33, 256)
        
        conv_outputs = conv_outputs[0, :, :, :]
        # print(conv_outputs.shape) --> (33, 33, 256)
        
        # make the conv_outputs dimension to have the channels first.
        # Roll the specified axis backwards, 
        # until it lies in a given position.
        conv_outputs = np.rollaxis(conv_outputs, 2)
        # print(conv_outputs.shape) --> (256, 33, 33)
        
        # get the weights pertaining to the predicted class.
        class_weights = self.model.layers[-1].get_weights()[0]
        # print(self.model.layers[-1]) --> Dense layer
        
        # Create the class activation map.
        cam = np.zeros(shape = conv_outputs.shape[1:3], dtype=np.float32)

        for idx, weight in enumerate(class_weights[:]):
            cam += weight * conv_outputs[idx, :, :]

        cam /= np.max(cam)
        cam = cv2.resize(cam, (image.shape[1], image.shape[2]))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # get the region of interest of the model, which part of the
        # image the model is looking at...
        heatmap[np.where(cam < 0.3)] = 0

        return heatmap
    

    def DisplayHeatMap(self, heatmap):
        
        '''
        Purpose: 
            Display the heatmap - the class level activations.

        Parameters:
            1. heatmap - The class activation heatmap for the concerned image.

        Return Value: 
            NONE
            
        '''
            
        # Display heatmap
        plt.matshow(heatmap)
        plt.axis('off')
        plt.show()



    def GetSuperImposedCAMImage(self, heatmap, image):
        
        '''
        Purpose: 
            Get the super imposed or the blended version of the image comprising of the class activations
            heatmap and the original image.

        Parameters:
            1. heatmap - The class activation heatmap for the concerned image.
            2. img - The original image for which the activation image is being calculated.

        Return Value: 
            superImposedImage - The blended version of the original image and the corresponding class 
                                activation map.
                                
        '''
            
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        
        # convert the image from BGR format to the RGB format.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Rescale the image
        image = np.uint8(255 * image)
       
        superImposedImage = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0.0)
        
        return superImposedImage

    
    def DisplaySuperImposedImages(self, image, heatmap, superimposed_img):
        
        '''
        Purpose: 
            Display the original image and the corresponding class activation blended image.

        Parameters:
            1. image - The original image for which the activation image is being calculated.
            2. superimposed_img - The blended version of the original image and the corresponding class 
                                  activation map. 

        Return Value: 
            NONE
            
        '''
        
        # Convert the image from BGR format to the RGB format.
        # Rescale the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.uint8(255 * image)
        
        fig, ax = plt.subplots(1, 3, figsize=(8, 12))

        ax[0].imshow(image)
        ax[1].imshow(heatmap)
        ax[2].imshow(superimposed_img)

        ax[0].title.set_text('Original Image')
        ax[1].title.set_text('Class\nActivation\nHeatmap')
        ax[2].title.set_text('Class\nActivation\bBlended\nImage')

        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')