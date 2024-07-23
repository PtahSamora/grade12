import streamlit as st
from PIL import Image, ImageOps, ImageFilter , Image
# from Image import Resampling
import pytesseract
import numpy as np


def load_image(image_file):
    img = Image.open(image_file)
    return img

def preprocess_image(img):
    # Convert to grayscale
    img = ImageOps.grayscale(img)
    
    # Apply adaptive thresholding
    # img = img.filter(ImageFilter.MedianFilter())
    # threshold = np.array(img).mean()  # Find average brightness
    # img = img.point(lambda p: p > threshold and 255)  # Simple thresholding to make image binary
    
    # Resize for better OCR accuracy
    base_width = 1000  # Resize width
    w_percent = (base_width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
    
    # Additional filters can be applied if needed
    # img = img.filter(ImageFilter.SHARPEN)

    return img

def extract_text_from_image(img):
    result_text = pytesseract.image_to_string(img)
    return result_text