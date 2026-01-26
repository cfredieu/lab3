# tests/test_model.py
import pytest
import torch
from src.retrieval.model import ModelService, Net


def test_net_has_correct_architecture():
    net = Net()
    assert isinstance(net.conv1, torch.nn.Conv2d)
    assert net.conv1.out_channels == 16
    assert net.conv1.kernel_size == (5, 5)

    assert isinstance(net.conv2, torch.nn.Conv2d)
    assert net.conv2.out_channels == 16

    assert net.fc1.in_features == 16 * 5 * 5
    assert net.fc3.out_features == 10


def test_model_service_initialization():
    from src.retrieval.main import model_service   # import the patched global
    assert model_service is not None
    assert model_service.model is not None
    assert model_service.device.type in ("cpu", "cuda")
    assert len(model_service.classes) == 10
    assert "plane" in model_service.classes
    assert "truck" in model_service.classes


def test_preprocess_image_shape(sample_image_bytes):
    from src.retrieval.main import model_service
    tensor = model_service.preprocess_image(sample_image_bytes)
    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.device == model_service.device
    assert tensor.min() > -2.0
    assert tensor.max() < 2.0


@pytest.mark.asyncio
async def test_predict_returns_valid_result(sample_image_bytes):
    from src.retrieval.main import model_service
    pred_class, confidence, top5 = model_service.predict(sample_image_bytes)
    assert isinstance(pred_class, str)
    assert pred_class in model_service.classes
    assert 0.0 <= confidence <= 1.0
    assert len(top5) == 5
    assert all(isinstance(c, str) and 0 <= p <= 1 for c, p in top5)
    probs = [p for _, p in top5]
    assert abs(sum(probs) - 1.0) < 0.01