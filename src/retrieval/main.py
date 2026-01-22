"""
Lab 3 FastAPI API.
@author: Kevin Lundeen
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 0.1.0+w26
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    message: str


# Define lifespan function to load models on startup
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Code before the 'yield' is executed during application startup
    try:
        logger.info("Loading models...")
        # FIXME: Load your model(s) here
        logger.info("Model loaded successfully!")
    except Exception as e:
        # Don't crash the server, but log the error
        logger.error(f"Failed to load model: {str(e)}")

    yield  # The application starts receiving requests after the yield

    # Code after the 'yield' is executed during application shutdown
    logger.info("Application shutting down (lifespan)...")


# Initialize FastAPI app
app = FastAPI(
    title="FIXME: API Title",
    description="FIXME: API Description",
    version="0.1.0",
    lifespan=lifespan,
)

# Add cross-origin resource sharing (CORS) middleware
# (gives browser permission to call our API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Implement health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check if the API is running.

    Returns:
        Health status
    """
    return HealthResponse(status="healthy", message="API is running and ready")


# Add error handler for general exceptions
@app.exception_handler(Exception)
async def general_exception_handler(_request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Create a test endpoint that raises exceptions (only for testing!)
@app.get("/test/error")
async def test_error():
    raise RuntimeError("Something went wrong")


# Mount static files LAST - catches all remaining routes
# including / --> /static/index.html, and
#           /stlye.css --> /static/style.css
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("To run this application:")
    print("uv run uvicorn src.retrieval.main:app --reload")
    print("\nThen open: http://localhost:8000")
