import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Define the CNN (same as in train_model.py)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(self._get_conv_output_size(), 128)
        self.fc2 = nn.Linear(128, 2)

    def _get_conv_output_size(self):
        # Create a dummy input tensor to calculate the size of the output of the convolutional layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 128)
            dummy_output = self._forward_conv(dummy_input)
            return int(np.prod(dummy_output.size()))

    def _forward_conv(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Define transformation for the input image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Function to classify an image
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Convert image to RGB
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Ensure the image path is correct
image_path = 'data/train/no/Screenshot 2024-07-24 105638.png'  # Use a specific test image
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# Example usage
if __name__ == "__main__":
    is_discounted = classify_image(image_path)
    if is_discounted == 1:
        print("Product has a discount.")
    else:
        print("Product does not have a discount.")
