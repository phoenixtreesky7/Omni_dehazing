# Omni_dehazing


# Stripe Sensitive Convolution for Omnidirectional Image Dehazing

## Abstract
The haze in a scenario may affect the 360 photo/video quality and the immersive 360◦ Virtual Reality (VR) panoramas experience. Recent single image dehazing methods up to now have been only focused on the plane images. In this work, we propose a novel neural network pipeline for single omnidirectional image dehazing. To achieve this, we built the first omnidirectional hazy image dataset, which contains both synthetic and real-world samples. We propose a new stripe sensitive convolution (SSConv) to handle the distortion problems due to the equirectangular projection. The SSConv calibrates distortion by two steps: 1) extracting features using different rectangle filters and 2) learning to select the optimal features by weighting to feature stripes (a series of rows in the feature maps). Subsequently, using the SSConv, we propose an end-to-end network that jointly learns haze removal and depth estimation from a single omnidirectional image. The estimated depth map is leveraged as the intermediate representation and provides global context and geometric information to the dehazing module. Extensive experiments on the challenging synthetic and real-world omnidirectional image datasets demonstrate the effectiveness of SSConv, and our network obtains superior dehazing performance. The experiments on practical applications also demonstrate that our method can significantly improve the 3D object detection and 3D layout performances for the omnidirectional hazy images.


# OmniHaze Datasets
The OmniHaze Datasets can be obtain from:

Synthetic samples:

link：https://pan.baidu.com/s/1buo0aPjZqFh5GwVg2Nueuw 

password：omdh 

Real-world samples:

link：https://pan.baidu.com/s/1nTnrDngpWqGNJOyYnNEgCg 

password：omdh 


