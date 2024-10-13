import os
from django.conf import settings
from datetime import datetime, timezone, timedelta
from celery import Celery

import pandas as pd
import redis
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
import environ
import requests
from io import BytesIO
import base64
import uuid
import subprocess

env = environ.Env()


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kiosk.settings")
app = Celery("kiosk")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

def send_image_to_api(image_b64, name, emp_id):
    """
    Sends an image to the 'upload_transaction_image' API endpoint using the requests library.
    """
    url = 'http://sonynginx/api/upload_transaction_image'  # API endpoint
    
    # Decode the Base64 string to bytes
    image_bytes = base64.b64decode(image_b64)

    # Generate a unique filename
    image_file_path = f'/tmp/image_{uuid.uuid4()}.jpg'
    
    # Write the bytes to a JPEG file in a temporary location
    with open(image_file_path, 'wb') as f:
        f.write(image_bytes)

    # Prepare the payload for the POST request
    data = {
        'emp_id': emp_id,  # Example emp_id, update as needed
        'name': name,
    }

    try:
        # Send the POST request with the image and additional data
        with open(image_file_path, 'rb') as image_file:  # Open the saved file as a binary file
            files = {'image': image_file}
            response = requests.post(url, files=files, data=data)
        
        if response.status_code == 201:
            print("Image uploaded successfully!")
        else:
            print(f"Failed to upload image. Status code: {response.status_code}, Error: {response.text}")

    except Exception as e:
        print(f"Error while sending image to API: {str(e)}")

    finally:
        # Clean up by removing the temporary image file
        if os.path.exists(image_file_path):
            os.remove(image_file_path)  # Remove the temporary file
            print(f"Temporary file {image_file_path} deleted.")
            response_data = response.json()
            return response_data


def redis_add_transaction(emp_id, location_id):
    r = redis.Redis(host='sony-redis', port=6379, password='2vBuMI9QeQ9tMGeG', db=0)
    
    # Create a unique key for each transaction (e.g., using emp_id and location_id)
    key = f'transaction_{emp_id}_{location_id}'
    value = f'{emp_id}_{location_id}'
    
    # Set the value with a TTL of 5 minutes (300 seconds)
    r.set(key, value, ex=300)  # 300 seconds = 5 minutes

def redis_check_transaction(emp_id, location_id):
    r = redis.Redis(host='sony-redis', port=6379, password='2vBuMI9QeQ9tMGeG', db=0)
    
    # Create the same unique key as used when adding the transaction
    key = f'transaction_{emp_id}_{location_id}'
    
    # Check if the key exists in Redis
    if r.exists(key):
        return False  # Transaction already exists
    else:
        return True  # Transaction does not exist

def add_transaction(emp_id, location_id, image_path):
    url = 'http://sonynginx/api/transactions'
    
    data = {
        "emp_id": emp_id, 
        "location_id": location_id,
        "image": f"http://localhost:81{image_path}"
    }
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)

    if response.status_code == 201:
        redis_add_transaction(emp_id, location_id)
        print("Transaction successfully added!")
    else:
        print(f"Failed to add transaction: {response.status_code} - {response.text}")

#Test connect celery
@app.task(bind=True)
def test(self):
    print("Hello ,world testing by minstone 2222")

#face_prediction function
@app.task(bind=True)
def face_recognition(self, dataset, input_vector, transaction_image):
    similarities = []

    # Iterate through the list of dictionaries (dataset)
    for row in dataset:
        embedding = row['embedding']  # Access the embedding directly from the dictionary
        # Compute cosine similarity between input vector and row's embedding
        similarity = pairwise.cosine_similarity([embedding], [input_vector])
        similarities.append(similarity[0][0])

    # Convert dataset back into a DataFrame to handle sorting
    df = pd.DataFrame(dataset)
    df['cosine'] = similarities

    # Sort the DataFrame by cosine similarity
    df_sorted = df.sort_values(by='cosine', ascending=False)

    # Get the top match
    top_match = df_sorted[['name', 'cosine', 'emp_id']].iloc[0]

    # Check if the similarity is above the threshold
    same_face_check = redis_check_transaction(top_match['emp_id'], env('LOCATION_ID'))
    if top_match['cosine'] >= 0.5 and same_face_check:
        print(f"Best match: {top_match['name']} with similarity: {top_match['cosine']:.4f} with emp_id: {top_match['emp_id']}")
        response = send_image_to_api(transaction_image, top_match['name'], top_match['emp_id'])
        emp_id = response.get('emp_id')
        image = response.get('image')
        location_id = env('LOCATION_ID')
        add_transaction(emp_id, location_id, image)
        
    else:
        print(f"No match found, ===> {top_match['cosine']}")