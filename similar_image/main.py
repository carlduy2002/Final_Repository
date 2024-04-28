# import shutil
# from PIL import Image
# from io import BytesIO
# from email.mime.multipart import MIMEMultipart
# from email.mime.image import MIMEImage
# import numpy as np
# from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from typing import List
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import base64
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, FilePath

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class FilePath(BaseModel):
    filePath: str

#Function to get image file
@app.post("/process_file")
async def process_file(file_path: str = Form(...)):
    #Resize the image file
    target_image = load_image(file_path)

    #Folder contain the images
    data_folder = "D:\Learning\TopUp\Graduation_Project_(COMP1682)\Term 2\web_project_BE\web_project_BE\image"

    #Perform get image file and the image in folder data_folder to perform compare
    similarity_scores = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_path = os.path.join(root, file)
                compare_image = load_image(image_path)
                similarity_score = calculate_cosine_similarity(target_image, compare_image)
                similarity_scores.append((image_path, similarity_score))

    #Sort the similarity 
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_similar_images = similarity_scores[1:5]

    #Extract paths of the top 3 similar images
    top_similar_image_paths = [image_path for image_path, _ in top_similar_images]

    return {'message' : top_similar_image_paths}

#Function to load image
def load_image(path_image):
    #Read image
    image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
    # Resize image to a consistent shap
    resized_image = cv2.resize(image, (244, 244))
    return resized_image

#Function to measure the similarity
def calculate_cosine_similarity(image1, image2):
    #Get the first image
    flat_image1 = image1.flatten().reshape(1, -1)
    #Get the second image
    flat_image2 = image2.flatten().reshape(1, -1)
    #Calculate the similar by the cosine_similarity of sklearn.metrics.pairwise library
    similarity = cosine_similarity(flat_image1, flat_image2)
    return similarity[0][0]
    


