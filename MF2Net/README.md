# MF2Net (A multi-level visual feature fusion method) 

This repository contains a complete pipeline for industrial defect image classification using a multi-level feature fusion approach. The system integrates contour-based structural features (Hu Moments) and deep semantic features (EfficientNet), followed by classification using SVM and Random Forest with a fusion decision strategy.

## Modules

1. **Image Upload** - Load local image samples.
2. **Preprocessing** - Convert to grayscale and apply Gaussian blur.
3. **Edge & Hu Moments** - Enhance edges and extract Hu moment features.
4. **Convolutional Embeddings** - Extract high-level features using EfficientNet.
5. **Classification & Fusion** - Perform classification with SVM and RF, then fuse predictions.
6. **Main Controller** - Integrate all steps in a single script.

## Dataset

Due to confidentiality concerns and the risk of technology leakage related to manufacturing companies, the full dataset is not publicly available. Only a few example images used in the published paper are included for demonstration purposes.

Researchers who are interested in validating the proposed method or conducting related academic studies are welcome to contact the authors to discuss potential data access under appropriate agreements or collaborative frameworks.

