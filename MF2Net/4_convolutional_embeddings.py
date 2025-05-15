import numpy as np
import cv2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model

# Load pre-trained EfficientNetB0 model (exclude top layer)
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def resize_and_preprocess(image, target_size=(224, 224)):
    """
    Resize and preprocess the image for EfficientNet input.
    """
    image_resized = cv2.resize(image, target_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_array = np.expand_dims(image_rgb, axis=0)
    return preprocess_input(image_array)

def extract_deep_features(image):
    """
    Extract convolutional embeddings using EfficientNetB0.
    """
    preprocessed = resize_and_preprocess(image)
    features = model.predict(preprocessed)
    return features.flatten()
