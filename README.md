# ImageSegmentation
Image segmentation on GrapesNet datset
## GrapesNet Dataset
GrapesNet dataset has been proposed which includes 4 different datasets of grape bunches of vineyard and are listed below:
1. Dataset-1: RGB images with one grape bunch per image with black background and one
grape bunch per image with natural background
2. Dataset-2: RGB images with two or more grape bunches per image with natural background
3. Dataset-3: RGB-D images of vineyard with real environmental conditions
4. Dataset-4: RGB-D images with one grape bunch per image with experimental environment
with coral background for grape weight prediction task

Dataset link
<link>https://data.mendeley.com/datasets/mhzmzd5cwx/1</link>

## Models

### Segnet

paper link: 

SegNet is a deep learning architecture designed for semantic segmentation tasks in computer vision. It was introduced as a way to perform pixel-wise classification in images, where each pixel is assigned a class label to identify objects or regions within the image. Here's a brief overview of SegNet:

1. **Encoder-Decoder Architecture**: SegNet follows an encoder-decoder architecture. The encoder is responsible for extracting features from the input image, while the decoder reconstructs the segmented output.

2. **Convolutional Neural Networks (CNNs)**: Both the encoder and decoder are composed of multiple layers of convolutional neural networks. The encoder uses convolutional layers to progressively reduce the spatial dimensions of the input image while extracting hierarchical features. The decoder, on the other hand, upsamples the low-resolution feature maps to generate a high-resolution segmentation map.

3. **Pooling Indices**: A distinctive feature of SegNet is its use of max-pooling indices during the downsampling (pooling) operation in the encoder. Instead of pooling layers simply storing the maximum values, SegNet retains the indices of the maximum values. These indices are later used in the decoder during upsampling to perform precise pixel-wise classification.

4. **Skip Connections**: To improve segmentation accuracy, SegNet incorporates skip connections that connect corresponding layers in the encoder and decoder. These connections help in preserving fine-grained details during the upsampling process.

5. **Softmax Layer**: The final layer of the SegNet decoder typically includes a softmax activation function to produce class probability scores for each pixel. These scores determine the pixel's class label, and the class with the highest score is assigned to each pixel.

6. **Applications**: SegNet has been used in various applications, including autonomous driving for road and scene understanding, medical image analysis for organ segmentation, and object detection in robotics and surveillance.

7. **Advantages**: SegNet's architecture allows for efficient pixel-wise classification, making it suitable for real-time or near-real-time applications. Its use of max-pooling indices and skip connections helps maintain spatial information and improve segmentation accuracy.

8. **Limitations**: While SegNet performs well in many segmentation tasks, it may struggle with objects of varying scales and complex scene understanding. More advanced architectures, such as U-Net and DeepLab, have been developed to address some of these challenges.

In summary, SegNet is an encoder-decoder architecture designed for semantic segmentation tasks, with a unique focus on preserving spatial information using max-pooling indices and skip connections. It has found applications in various fields where accurate pixel-wise classification is essential.

## Results using Segnet 
### Input image

![Alt Text](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/input.png)

### Output image

![Alt Text](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/segnet.png)

## ResNet50 

A ResNet-50 based PSPNet (Pyramid Scene Parsing Network) is a deep learning architecture used for image segmentation tasks. It combines the power of two popular deep learning architectures: ResNet-50 for feature extraction and PSPNet for scene parsing. Here's a brief overview of this combination:

1. **ResNet-50 as the Backbone**:
   - ResNet-50 is used as the backbone network for feature extraction. ResNet (Residual Network) is known for its ability to train very deep neural networks effectively while mitigating the vanishing gradient problem. ResNet-50 is a specific variant of ResNet that consists of 50 layers.

2. **Feature Extraction**:
   - The ResNet-50 backbone processes the input image and extracts hierarchical features at different spatial scales. These features capture both low-level and high-level visual information.

3. **PSPNet Module**:
   - PSPNet, or Pyramid Scene Parsing Network, is a module added on top of the ResNet-50 backbone. Its primary purpose is to capture contextual information at multiple scales, which is crucial for accurate image segmentation.
   - PSPNet uses a pyramid pooling module, which involves spatial pyramid pooling (SPP) at different levels. This module captures context information at various receptive field sizes.
   - The PSPNet module helps the network understand the global context of the scene and enables it to make more informed pixel-wise predictions.

4. **Semantic Segmentation**:
   - The final layers of the ResNet-50 based PSPNet architecture are typically dedicated to semantic segmentation. These layers take the features extracted by the ResNet-50 backbone and the contextual information captured by the PSPNet module to produce a pixel-wise segmentation map.
   - The segmentation map assigns a class label to each pixel in the input image, identifying objects or regions of interest.

5. **Applications**:
   - ResNet-50 based PSPNet architectures are widely used in computer vision tasks that require accurate semantic segmentation, such as autonomous driving, medical image analysis, scene understanding, and more.
   - They are particularly effective when the context and spatial relationships between objects in the scene are essential for accurate segmentation.

6. **Advantages**:
   - Combining ResNet-50 with PSPNet provides a powerful solution for image segmentation. It benefits from ResNet's strong feature extraction capabilities and PSPNet's ability to capture contextual information.
   - The architecture can handle complex scenes with multiple objects and varying scales.

7. **Challenges**:
   - Training and deploying deep networks like ResNet-50 based PSPNet can be computationally intensive and require significant GPU resources.
   - Adequate labeled data for segmentation tasks is crucial for training a robust model.

In summary, a ResNet-50 based PSPNet is a deep learning architecture that leverages the ResNet-50 backbone for feature extraction and adds the PSPNet module for capturing contextual information. This combination is effective for accurate semantic image segmentation and is used in various computer vision applications.

## Results using ResNet50 and PSPnet

### Input image

![Alt Text](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/input.png)

### Output image

![Alt Text](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/resnet%2050%20segmented.png)

## Vgg_unet

## Results using Vgg_unet

### Input image

![Alt Text](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/input.png)

### Output image

![Alt Text](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/vgg_unet.png)

## Detectron

**Backbone Convolutional Neural Network (CNN)**:
At the core of Mask R-CNN is a backbone CNN, such as ResNet or VGG. This backbone network is responsible for extracting meaningful features from the input image. Think of it as the part of the network that understands the basic patterns, shapes, and structures in the image.
Region Proposal Network (RPN):

This component takes the feature maps generated by the backbone CNN and proposes potential regions of interest in the image. These regions are bounding boxes that might contain objects. The RPN evaluates how likely it is that each region contains an object.
RoI Align (Region of Interest Align):

RoI Align takes the proposed regions (bounding boxes) and extracts fixed-size feature maps from the backbone CNN. Unlike traditional methods, RoI Align ensures that the extracted features align precisely with the object boundaries, preserving spatial details.
Object Classification Head:

After feature extraction, Mask R-CNN has a network head that performs object classification. It predicts what each proposed region contains (e.g., "cat," "dog," "car") and how confident it is in this prediction.
Bounding Box Regression Head:

This part refines the bounding boxes proposed by the RPN, making them more accurate by adjusting their positions and sizes.
Mask Generation Head:

The unique feature of Mask R-CNN is its ability to generate masks for each object within the proposed regions. This head is responsible for predicting pixel-level masks that outline the shape of each object.
Combination and Output:

The outputs of the object classification, bounding box regression, and mask generation heads are combined to provide the final results. For each object, you get:
The class label (e.g., "cat").
The refined bounding box that tightly fits the object.
A detailed pixel-level mask outlining the object's shape.

## Results using Detectron-2


![Alt Text](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/WhatsApp%20Image%202023-09-20%20at%2015.27.04.jpg)

## FCN

FCN, or Fully Convolutional Network, is a deep learning architecture designed for pixel-level image segmentation.

It starts with a feature extractor like VGG or ResNet and replaces the fully connected layers with convolutional layers to maintain spatial information.

Upsampling layers increase the resolution of feature maps, and skip connections fuse high and low-level features to improve accuracy. The final convolutional layer produces pixel-wise class predictions.

## Results usinf FCN
![Alt Text](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/fcn.jpg)



### How to perform Image segmentation on your own dataset

If you have your dataset on which you want to perform image segmentation first you will need annotated images for that. 

### How to get annotated images?

There are two ways to achieve this:
1. By using edge detection techniques. Colab file link : 

2. The manual method : 
You can use various image annotation softwares. One such is VGG image annotator (VIA) by Oxford university. 
Video tutorial for VIA : <a href = "https://www.youtube.com/watch?v=AYLJ3YC07oI">Click Me!</a> 
The above video does not have the code s here's a colab file for speeding things up : <a href = "https://colab.research.google.com/drive/12XxXG-ZQryopv-resY6VUa5QGjnD_hJc?usp=sharing>Link</a>

### Preparing dataset in batches using Imagetraingenerator in tensorflow


### Implementations of various semantic segmentation algorithms and their results:

