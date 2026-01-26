# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
import src.retrieval.main
from src.retrieval.main import app, MODEL_PATH
from src.retrieval.model import ModelService
import os


@pytest.fixture(scope="session", autouse=True)
def _load_model_for_tests():
    """
    Force-load the model once per test session and patch the global in main.py.
    autouse=True makes it run automatically for all tests.
    """
    print("Session fixture: Loading model for all API tests...")
    print(f"Using MODEL_PATH: {MODEL_PATH}")
    print(f"File exists? {os.path.isfile(MODEL_PATH)}")

    try:
        loaded_model = ModelService(MODEL_PATH)
        src.retrieval.main.model_service = loaded_model  # ← explicitly set the global
        print("Session fixture: Model loaded and global patched successfully!")
    except Exception as e:
        print(f"Session fixture: Model loading FAILED: {str(e)}")
        pytest.exit(f"Cannot run API tests: model failed to load - {str(e)}", returncode=1)

    yield  # tests run here

    # Optional: cleanup
    src.retrieval.main.model_service = None
    print("Session teardown: model_service reset")


@pytest.fixture(scope="module")
def test_app(_load_model_for_tests):  # depend on the session fixture
    """Test client – model is already loaded via session fixture"""
    client = TestClient(app)
    # Debug check
    print(f"test_app fixture: model_service loaded? {src.retrieval.main.model_service is not None}")
    return client


@pytest.fixture
def sample_image_bytes():
    test_img_path = os.path.join(os.path.dirname(__file__), "automobile10.png")
    if not os.path.exists(test_img_path):
        pytest.skip(f"Missing test image at: {test_img_path}")
    with open(test_img_path, "rb") as f:
        return f.read()
