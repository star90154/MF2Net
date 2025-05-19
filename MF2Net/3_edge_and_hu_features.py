import cv2
import numpy as np

def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Perform Canny edge detection.
    """
    return cv2.Canny(image, low_threshold, high_threshold)

def sobel_edge_detection(image, ksize=3):
    """
    Perform Sobel edge detection and return combined edge map.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = cv2.magnitude(grad_x, grad_y)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    return sobel

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

def extract_combined_features(gray_img, edge_type='canny'):
    """
    Extract edge map and Hu moments from the image using selected edge method.
    """
    if edge_type == 'canny':
        edge = canny_edge_detection(gray_img)
    elif edge_type == 'sobel':
        edge = sobel_edge_detection(gray_img)
    else:
        raise ValueError("Unsupported edge_type: choose 'canny' or 'sobel'")

    enhanced_edge = apply_morphological_operations(edge)
    hu = extract_hu_moments(enhanced_edge)
    return hu
