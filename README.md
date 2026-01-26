# Lab 3: CIFAR-10 CNN Classifier & Model Service API  
 
**ARIN 5360 – AI Systems**  
Semantic image classification system using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.  
The project provides a FastAPI-based REST API for model inference, health checks, and a simple web frontend.

## Features

- CNN architecture trained on CIFAR-10 (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- FastAPI backend with automatic OpenAPI docs (Swagger UI)
- Model loaded once at startup (global singleton pattern)
- Input image preprocessing (resize, normalize, batch)
- Top-1 + top-5 predictions with confidence scores
- Basic static frontend (HTML + CSS) for uploading images
- Comprehensive test suite (unit + API + smoke tests) using pytest
- Model file: `gg_classifier.pt` (PyTorch state dict)  

## Quick Start  

### 1. Prerequisites

```bash
- Python 3.10–3.13
- [uv](https://github.com/astral-sh/uv) (recommended) or pip + virtualenv  
``` 

### 2. Installation  

```bash
# Clone the repository (if not already done)
# git clone <your-repo-url> 
# cd lab3 

# Install dependencies  
uv sync 
``` 
### 3. Run the API Server  

```bash
# Start server  
uv run uvicorn src.retrieval.main:app --reload
```
Server starts at url http://localhost:8000   

#### 4. Try it Out

```bash
- Open browser and go to http://localhost:8000  
- Upload any image via the simple web interface. 
```

## Project Structure  
```
lab3
├── documents
│   └── sample1.txt
├── pyproject.toml          # Project dependencies
├── README.md
├── src
│   └── retrieval
│       ├── __init__.py
│       └── main.py         # FastAPI app, routes, model loading
│       └── model.py        # ModelService, Net (CNN)
├── static                  # Frontend
│   ├── index.html
│   └── style.css
├── tests                   # Tests
│   ├── __init__.py
│   └── automobile10.png    # Sample test image
│   └── conftest.py         # Shared fixtures (model loading, test client)
│   └── test_api.py         # API endpoint test
│   └── test_model.py       # Model logic & inference tests
│   └── test_smoke.py       # Smoke and health tests
└── gg_classifier.pt        # Trained model weights
└── uv.lock


