'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
DESCRIPTION: Grad-CAM class activation visualization.
             A Utility/Helper class that contains helper methods for visualizing the class level activations of 
             a Convolotional Neural Network. Grad-CAM visualizations help us know which regions, patterns of 
             the image the neural network is looking at, and activating the region around those patterns. To 
             visualize the activation maps, we would need the output of the LAST CONVOLUTIONAL LAYERS
             and the final CLASSIFICATION LAYERS.
             
REFERENCE:   (François Chollet, 2020)
             François Chollet, (2020) Grad-CAM class activation visualization. 
             [online] Available at: https://keras.io/examples/vision/grad_cam/ [Accessed 20 Jun. 2020].

WEBSITE:     https://keras.io/examples/vision/grad_cam/
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Import Libraries
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

# Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class GradCamUtils():
    
    '''
    A Utility/Helper class that contains helper methods for visualizing the gradient of the 
    class level activations of a Convolotional Neural Network. CAM visualizations help us know which 
    regions, patterns of the image the neural network is looking at, and activating the region around 
    those patterns. To visualize the activation maps, we would need the output of the LAST CONVOLUTIONAL 
    LAYERS.
    
    Args:
        model (keras model): the classification model.
        
        layerName (string): name of the layer for which the
                   class activations are to be visualized.
    
    '''
    
    def __init__(self, model, last_conv_layer_name):
        
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        
        # get the layer object of the last convolutionl layer
        last_conv_layer = self.model.get_layer(self.last_conv_layer_name).output
        
        # create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        self.grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])


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


    def compute_heatmap(self, img_array, pred_index=None):
        
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
            4. pred_index (optional): The index of the largest value in the tensor. 
                                      The unit which contributed the most towards the predicted result.

        Return Value: 
            heatmap - The heatmap showing the more active regions the CNN looked at in deciding the class
                      for the image.
                      
        '''
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0]) # returns the index of the largest value in the tensor.
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel.
        # get the mean of elements across dimensions of a tensor.
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        return heatmap.numpy()
    
    

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


    
    def GetSuperImposedCAMImage(self, heatmap, image, alpha = 0.5):
        
        ''''
        Purpose: 
            Get the super imposed or the blended version of the image comprising of the class activations
            heatmap and the original image.

        Parameters:
            1. heatmap - The class activation heatmap for the concerned image.
            2. image - The original image for which the activation image is being calculated.
            3. alpha (optional) - A number between 0 & 1 to control the weighted blending of the images.

        Return Value: 
            superImposedImage - The blended version of the original image and the corresponding class 
                                activation map.

        '''
        
        # Rescale the image to 0-255
        image = image * 255

        # Rescale heatmap to 0-255
        # Resize to the size of the image
        # Apply colormap
        heatmap = np.uint8(255 * heatmap)
        
        # upsample the class actiavtion to match the size of the original image 
        # before the 2 images can be blended for final visualization.
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Superimpose the heatmap on original image
        superImposedImage = heatmap * alpha + image
        superImposedImage = tf.keras.preprocessing.image.array_to_img(superImposedImage)
        
        return superImposedImage

    
    def DisplaySuperImposedImages(self, image, heatmap,superimposed_img):
        
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
        
        fig, ax = plt.subplots(1, 3, figsize=(8, 12))

        ax[0].imshow(image)
        ax[1].imshow(heatmap)
        ax[2].imshow(superimposed_img)

        ax[0].title.set_text('Original\nImage')
        ax[1].title.set_text('Class Activation\nHeatmap')
        ax[2].title.set_text('Class Activation\nBlended Image')

        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')