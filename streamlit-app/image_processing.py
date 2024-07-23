import pytesseract
import cv2
import cv2 as cv
from PIL import Image
import numpy as np
from openai import OpenAI
import os

from dotenv import load_dotenv
# import doctr 
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
# from doctr.models import ocr_predictor



load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


client = OpenAI()

def load_image(image_file):
    """Load an image from a file uploader."""
    return Image.open(image_file)

def preprocess_image(img):
    """Use OpenCV for preprocessing the image to enhance OCR accuracy."""
    load_img = np.array(img.convert('L'))
    # load_img = cv.imread(img)
    # load_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # Apply Gaussian Blur to smooth out the image
    gaus_img = cv.GaussianBlur(load_img, (5, 5), 0)

    # Apply adaptive thresholding to get a binary image
    adthresh_img = cv.adaptiveThreshold(gaus_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY, 11, 2)

    # Invert image to have white text on black background, if necessary
    invert_img = cv.bitwise_not(adthresh_img)

    return gaus_img


# def extract_text_from_image(img):
#     """Extract text from the preprocessed image using pytesseract."""
#     result_text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
#     return result_text


def extract_text_from_image(preprocessed_img):
    # Convert OpenCV image format to PIL format
    img = Image.fromarray(preprocessed_img)

    # Configure tesseract to handle sparse text and use the LSTM OCR engine
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(img, config=custom_config, lang='eng')

    return extracted_text

def process_text_with_llm(text):
    try:
        response = client.chat.completions.create(model="gpt-4o-2024-05-13",
        messages = [{'role': 'system', 'content':"Act as a "},
                    {'role': 'user', 'content':f"{text}"}],
        temperature=0.1   
        )
        #         return 
        # response = client.completion.create(
        #     engine="gpt-4o-2024-05-13",  
        #     prompt=text,
        #     max_tokens=150
        # )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# class ExtractText:
#     """Detect and recognize texts from images."""
#     def __init__(self):
#         self.ocr_model = ocr_predictor(pretrained=True)
#     def __call__(self, batch):
#         img = DocumentFile.from_images(batch["path"]) # load images
#         text = self.ocr_model(img) # extract texts
#         return {"source": batch["path"], "text": [text.render()]}