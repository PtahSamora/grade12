import openai
import requests
from PIL import Image
import io
import os 
from dotenv import load_dotenv

load_dotenv()

# Your OpenAI API key
api_key = os.getenv('OPENAI_API_KEY') 

# Function to convert image to bytes
def image_to_bytes(image_path):
    with open(image_path, 'rb') as image_file:
        return image_file.read()

# Function to create chat completion with image
def create_chat_completion_with_image(api_key, image_path, prompt):
    image_data = image_to_bytes(image_path)

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    # Replace this URL with the actual endpoint if different
    url = 'https://api.openai.com/v1/chat/completions'

    files = {
        'image': ('image.jpg', image_data, 'image/jpeg'),
    }

    data = {
        'model': 'gpt-4',  # Or 'gpt-3.5-turbo' if using GPT-3.5
        'messages': [
            {'role': 'system', 'content': 'You are an AI that can analyze images and answer questions.'},
            {'role': 'user', 'content': prompt},
        ]
    }

    response = requests.post(url, headers=headers, files=files, json=data)
    return response.json()

# Example usage
image_path = '/Users/samorasixaba/Documents/Matric Preparation Project/test-images/quadratic-equation-formula-solution-of-solving-quadratic-equations-background-education-getting-grades-higher-school-math-programs-handwritten-math-text-grouped-and-isolated-on-white-free-vector.jpg'
prompt = 'A solution to a quadratic equation'

response = create_chat_completion_with_image(api_key, image_path, prompt)
print(response)
