import cv2
import numpy as np

def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Perform Canny edge detection.
    """
    return cv2.Canny(image, low_threshold, high_threshold)

def apply_morphological_operations(edge_img, kernel_size=3):
    """
    Apply dilation followed by erosion to enhance edges.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(edge_img, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded

def extract_hu_moments(image):
    """
    Compute Hu Moments from a binary or grayscale image.
    """
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments
