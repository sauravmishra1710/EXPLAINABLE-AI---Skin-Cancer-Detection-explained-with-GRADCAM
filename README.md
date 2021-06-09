# EXPLAINABLE-AI - Skin-Cancer-Detection-explained-with- CAM & GRADCAM

Diagnose the presence of skin cancer in a person using CNN and as well explain what led the CNN to arrive at the decision.  Visual explanations are made utilizing the Gradient-weighted Class Activation Mapping (Grad-CAM), the gradients flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for considered for arriving at the decision. The original paper for GRADCAM can be found @ https://arxiv.org/abs/1610.02391

# Explainable AI 

The application of AI systems in healthcare is a challenging task mainly because the factors involved in arriving at a decision by the machines are not explainable. Questions like, how did the machine arrive at this decision? or what did the machine see to predict the particular class of a condition? will always be asked to understand a machine’s way of taking the decisions in healthcare. 
Interpretability matters when machines take decisions on doctor’s behalf. For machines to arrive at a particular medical decision, health diagnostic or the treatment course, they have to be trustable. If machine based intelligent systems have to be integrated into the healthcare systems, their decisions have to be meaningful and transparent. The transparency on the thought process of such systems will help in the following ways - 

1.	When AI systems are weaker in the building phase, helps in building a robust system by identifying the failure reasons. 
2.	When AI systems are stable and achieved a certain SOTA result, transparency then will ensure that such systems are able to establish confidence on the healthcare personnel and the patients.
3.	When AI systems are powerful enough such that they are able to learn by their own experience, studying the machine’s ability of decision making may help healthcare professionals make better decisions themselves. 

# Papers Involved - 

As part of this study, I go through the implementations of 2 papers that try to explain what a CNN sees in an image before classifying it to a particular class. The 2 papers studied are -

1.  Learning Deep Features for Discriminative Localization. Available at - https://arxiv.org/abs/1512.04150
2.	Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. Available at - https://arxiv.org/abs/1610.02391

# CAM Overview

(Zhou et al., 2015) in their paper titled “Learning Deep Features for Discriminative Localization” utilize the Global Average Pooling layer to demonstrate its ability to support localization of objects in am image. Though, it has been majorly used for its regularizing capabilities to prevent overfitting, the authors of the paper say that GAP layer can also be used to retain the spatial structure of the feature maps and identify the discriminative regions of the image.
The output of the final convolutional layers are fed to the fully connected dense layers which results in a loss of the spatial structure. However, performing a global average pooling operation on the convolutional feature maps just before the final softmax layer, not only retains the spatial structure but also help identify the important regions in the image by projecting back the weights of the output layer onto the convolutional feature maps.

![CAM Building Blocks](https://github.com/sauravmishra1710/EXPLAINABLE-AI---Skin-Cancer-Detection-explained-with-GRADCAM/blob/main/images/CAM_Buildng_Blocks.PNG)

## CAM Visualizations

A few CAM viz from the experiment for visualizing the classification of skin cancer tumors as benign or malignant are shown in the images below

![CAM Viz 1](https://github.com/sauravmishra1710/EXPLAINABLE-AI---Skin-Cancer-Detection-explained-with-GRADCAM/blob/main/images/CAM_Viz1.PNG)

![CAM Viz 2](https://github.com/sauravmishra1710/EXPLAINABLE-AI---Skin-Cancer-Detection-explained-with-GRADCAM/blob/main/images/CAM_Viz2.PNG)

![CAM Viz 3](https://github.com/sauravmishra1710/EXPLAINABLE-AI---Skin-Cancer-Detection-explained-with-GRADCAM/blob/main/images/CAM_Viz3.PNG)

# GradCAM Overview

According to (Selvaraju et al., 2016), a good visual explanation should be able to localize the object of interest by capturing the minutest of details in the image. The Grad-CAM technique could explain a CNN’s decision making process answering, “why they predict what they predict”. It takes into consideration the gradients of the target object flowing into the final convolutional layer to create a rough localization mapping and highlighting the important regions of the target image the mode considered to assign the particular class. Grad-CAM relies on convolutional layer as they tend to retain the spatial information. It utilizes the gradient activations coming into the last convolutional layer as these layer look for more class specific features. 

![GradCAM Building Blocks](https://github.com/sauravmishra1710/EXPLAINABLE-AI---Skin-Cancer-Detection-explained-with-GRADCAM/blob/main/images/Compute_Gradients.PNG)

## GradCAM Visualizations

A few GradCAM viz from the experiment for visualizing the classification of skin cancer tumors as benign or malignant are shown in the images below

![GradCAM Viz 1](https://github.com/sauravmishra1710/EXPLAINABLE-AI---Skin-Cancer-Detection-explained-with-GRADCAM/blob/main/images/GradCAM_Viz1.PNG)

![GradCAM Viz 2](https://github.com/sauravmishra1710/EXPLAINABLE-AI---Skin-Cancer-Detection-explained-with-GRADCAM/blob/main/images/GradCAM_Viz2.PNG)

![GradCAM Viz 3](https://github.com/sauravmishra1710/EXPLAINABLE-AI---Skin-Cancer-Detection-explained-with-GRADCAM/blob/main/images/GradCAM_Viz3.PNG)



# References

1. Codella, N. et al. (2019) ‘Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)’. Available at: http://arxiv.org/abs/1902.03368 (Accessed: 27 May 2021).
2. Selvaraju, R. R. et al. (2016) ‘Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization’. doi: 10.1007/s11263-019-01228-7.
3. Tschandl, P., Rosendahl, C. and Kittler, H. (2018) ‘The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions’, Scientific Data, 5(1), p. 180161. doi: 10.1038/sdata.2018.161.
4. Zhou, B. et al. (2015) ‘Learning Deep Features for Discriminative Localization’. Available at: https://arxiv.org/abs/1512.04150 (Accessed: 7 June 2021).
