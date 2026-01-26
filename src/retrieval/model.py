"""
CNN Model Service for Image Classification
@author: Celine Fredieu
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 0.1.0+w26

STUDENTS:
1. Rename this file to model.py.
2. Work on the parts labeled FIXME. As you fix them remove the FIXME label.
"""
import io
import logging
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class Net(nn.Module):
    """CNN architecture exactly as the one that saved our model weights."""

    #Copy the contents of your Net class from L1 here exactly.
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModelService:
    """Service for loading model and running inference."""

    def __init__(self, model_path: str):
        """
        Initialize the model service.

        Args:
            model_path: Path to the saved model weights (.pt or .pth file)
        """
        #Set up ModelService object here
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.classes = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.load_model()  # last step in setting up

    def load_model(self):
        """Load the trained model from disk."""
        #Load the model weights from disk here
        try:
            self.model = Net()
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device))

        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        Preprocess image for model inference.

        Args:
            image_bytes: Raw image bytes from upload

        Returns:
            Preprocessed image tensor ready for the model
        """
        # Get the image into a form usable by the model here
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB") #(handles grayscale and RGBA)

        # Apply transformations and add batch dimension [1, C, H, W]
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze_(0)

        # Move to GPU
        image_tensor = image_tensor.to(self.device)
        return image_tensor


    def predict(self, image_bytes: bytes) -> Tuple[
        str, float, List[Tuple[str, float]]]:
        """
        Run inference on an image.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Tuple of (predicted_class, confidence, top_5_predictions)
            where top_5_predictions is a list of (class_name, probability)
            tuples
        """
        #Run inference on the image here
        if self.model is None:
            raise RuntimeError("Model not loaded")
        try:
            image_tensor = self.preprocess_image(image_bytes)
            with torch.no_grad():
                outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top5_prob, top5_indices = torch.topk(probabilities, k=5)

            #Convert to class names and probabilities
            top5_predictions = [(self.classes[idx], prob.item()) for idx,
                                prob in zip(top5_indices[0], top5_prob[0])]

            #Get the top prediction and return
            predicted_class = top5_predictions[0][0]
            confidence = top5_predictions[0][1]

            return predicted_class, confidence, top5_predictions

        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

# Example usage and testing
# (this code is automatically skipped when running the service)
if __name__ == "__main__":
    # Test your model service
    service = ModelService("./gg_classifier.pt")

    # Load a test image
    with open("./automobile10.png", "rb") as f:
        image_bytes = f.read()

    # Run prediction
    predicted_class, confidence, top5_ = service.predict(image_bytes)
    print(f"\nPrediction: {predicted_class}")
    print(f"Confidence: {confidence: .2%}")
    print("\nTop 5 predictions:")
    for class_name, prob in top5_:
        print(f" {class_name}: {prob:.2%}")
