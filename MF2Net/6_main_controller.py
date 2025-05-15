import os
from 1_upload_image import load_single_image
from 2_preprocessing import to_grayscale, apply_gaussian_blur
from 3_edge_and_hu_features import canny_edge_detection, apply_morphological_operations, extract_hu_moments
from 4_convolutional_embeddings import extract_deep_features
from 5_classification_fusion import FusionClassifier
import numpy as np

def process_image(image_path):
    image = load_single_image(image_path)
    gray = to_grayscale(image)
    blurred = apply_gaussian_blur(gray)
    
    edges = canny_edge_detection(blurred)
    enhanced_edges = apply_morphological_operations(edges)
    
    hu_features = extract_hu_moments(enhanced_edges)
    deep_features = extract_deep_features(image)
    
    combined = np.hstack((hu_features, deep_features))
    return combined

if __name__ == "__main__":
    # Example usage
    image_paths = ["sample1.jpg", "sample2.jpg"]
    labels = [0, 1]  # Placeholder labels

    X = []
    for path in image_paths:
        features = process_image(path)
        X.append(features)

    X = np.array(X)
    y = np.array(labels)

    clf = FusionClassifier()
    clf.fit(X, y)

    preds = clf.predict(X)
    print("Predictions:", preds)
