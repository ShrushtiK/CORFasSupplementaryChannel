# Enhancing CNN-based Image Classification Robustness using the Push-Pull CORF Operator for Shape-Biased Learning

**University of Groningen**  
**Master’s Thesis Project** by Shrushti Kaul (s5288843)  
Supervised by Prof. dr. G. Azzopardi & Sabatino Esposito  

[📄 View Colloquium Slides](./colloquium.pdf)

---

## Overview

Convolutional Neural Networks (CNNs), while highly effective on standard datasets, often fail under **challenging test conditions** such as corrupted, adversarial, or stylized images due to **texture bias**—relying on superficial features rather than shapes.

This project explores the **Push-Pull CORF (Combination Of Receptive Fields) model**, inspired by the visual cortex (V1), to enhance **shape-biased learning** in CNNs. Experiments use **ResNet-50** as the backbone with two augmentation techniques:

- **BlendAug:** Creates blended datasets of raw images and CORF contour maps to emphasize shape features.  
- **OnlineAug:** Efficiently samples blended datasets during training to maintain diversity without modifying the original data.

---

## Evaluation

Methods were tested on **ImageNet-C, ImageNet-A, and ImageNet-R**:

- **BlendAug:** Top-1 accuracy **4.2%** on ImageNet-A (vs 3.8% for AugMix)  
- **OnlineAug:** Top-1 error **58.99%** on ImageNet-R (matching AugMix 58.9%)

Results show that emphasizing shape features with **ResNet-50** improves robustness against adversarial and stylized inputs.

---

## Acknowledgments

Special thanks to Prof. dr. G. Azzopardi, S. Esposito, and the **University of Groningen Center for Information Technology**, including access to the **Habrók high-performance computing cluster**, which was essential for conducting the experiments.