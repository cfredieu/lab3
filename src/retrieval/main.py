"""
FastAPI application for CNN image classification.
@author: Celine Fredieu
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 0.1.0+w26

STUDENTS:
1. Rename this file to main.py.
2. Work on the parts labeled FIXME. As you fix them remove the FIXME label.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse

from .model import ModelService

# Our trained CNN model from the PyTorch tutorial
MODEL_PATH = "./gg_classifier.pt"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model service instance
model_service: Optional[ModelService] = None


# Define response models with Pydantic
class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predicted_class: str
    confidence: float
    top_5_predictions: List[dict]


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    message: str


# Define lifespan function to load model on startup
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Code before the 'yield' is executed during application startup

    global model_service

    try:
        logger.info("Loading CNN model...")
        logger.info(f"Attempting to load model from: {MODEL_PATH}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"File exists? {os.path.isfile(MODEL_PATH)}")
        # Initialize ModelService with the path to our trained CNN model.
        # Once at startup, not on every request! This is much more efficient.

        model_service = ModelService(MODEL_PATH)
        logger.info("Model loaded successfully!")

    except Exception as e:
        # Don't crash the server, but log the error
        logger.error(f"Failed to load model: {str(e)}")

    yield  # The application starts receiving requests after the yield

    # Code after the 'yield' is executed during application shutdown
    logger.info("Application shutting down (lifespan)...")


# Initialize FastAPI app
app = FastAPI(
    title="CNN Image Classifier API",
    description="Upload an image and get classification predictions",
    version="0.1.0",
    lifespan=lifespan,
)

# Add cross-origin resource sharing (CORS) middleware
# (gives browser permission to call our API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Implement health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check if the API is running and the model is loaded.

    Returns:
        Health status of the model loading state
    """
    # Check if the model is loaded
    if model_service is None:
        return HealthResponse(status="unhealthy", model_loaded=False, message="Model not loaded")
    return HealthResponse(
        status="healthy", model_loaded=True, message="API is running and model is ready"
    )


# Implement home page endpoint
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Serve the HTML upload interface.
    Returns:
        HTML page with upload form
    """
    # Read and return the HTML file
    # (remember to change it to static/index.html once that's ready)

    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>500 Error</h1><p>index.html not found in static/</p>"


# Implement prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and get a classification prediction.

    Args:
        file: Uploaded image file

    Returns:
        Prediction results with class and confidence

    Raises:
        HTTPException: If prediction fails
    """
    # Check if the model is loaded
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    # Validate their file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Call the model service to make a prediction
    try:
        # Read image bytes
        image_bytes = await file.read()

        # Run prediction
        predicted_class, confidence, top5 = model_service.predict(image_bytes)

        # Format response (turn it into something that makes sense)
        top5_formatted = [{"class": class_name, "probability": prob} for class_name, prob in top5]

        logger.info(f"Prediction: {predicted_class} ({confidence:.2%})")

        return PredictionResponse(
            predicted_class=predicted_class, confidence=confidence, top_5_predictions=top5_formatted
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Add error handler for general exceptions
@app.exception_handler(Exception)
async def general_exception_handler(_request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/test/error")
async def force_error():
    raise RuntimeError("This is a deliberate test error")


if __name__ == "__main__":
    print("To run this application:")
    print("\nuv run uvicorn main:app --reload")
    print("\nThen open: http://localhost:8000")
