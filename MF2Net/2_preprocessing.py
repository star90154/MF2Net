import cv2
import numpy as np

def to_grayscale(image):
    """
    Convert an image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    Apply Gaussian Blur to the image.
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)
