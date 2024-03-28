import uvicorn
from fastapi import FastAPI, File, UploadFile
import joblib
from PIL import Image
import numpy as np
from io import BytesIO
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500"],  # Allow requests from this origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# Load your pre-trained model
try:
 model = joblib.load("preloaded_model.pkl")
except Exception as e:
   print(e)

def process_image(image_bytes):
    # Load the image
    img = Image.open(BytesIO(image_bytes))

    # Preprocess the image (resize, convert to array, etc.)
    # Example:
    img = img.resize((224, 224))  # Resize to match model's input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array

@app.post("/predict/")
async def predict(image_file: UploadFile = File(...)):
    print('hey')
    # Read the image file as bytes
    image_bytes = await image_file.read()

    # Process the image
    processed_image = process_image(image_bytes)
    print('bbb')
    # Perform prediction using the loaded model
    prediction = model.predict(processed_image)

    # Optionally, you can return the prediction result
    return {"prediction": prediction.tolist()}
