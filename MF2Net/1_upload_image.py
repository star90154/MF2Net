import os
import cv2
from typing import List

def load_images_from_folder(folder_path: str, extensions: List[str] = ['.jpg', '.png', '.jpeg']) -> List:
    """
    Load all images from a specified folder with given extensions.
    """
    images = []
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in extensions):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img))
    return images

def load_single_image(image_path: str):
    """
    Load a single image from file.
    """
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        return img
    else:
        raise FileNotFoundError(f"Image not found: {image_path}")
