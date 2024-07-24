import os
import pyautogui
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import keyboard

# Define the CNN (same as in train_model.py)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1_input_dim = self._get_conv_output_size()
        self.fc1 = nn.Linear(self.fc1_input_dim, 128)
        self.fc2 = nn.Linear(128, 2)

    def _get_conv_output_size(self):
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

# Function to take a screenshot of the specified region
def screenshot_region(region, amount):
    x, y, width, height = region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    screenshot_path = f'data/temp_screenshot{amount}.png'
    screenshot.save(screenshot_path)
    return screenshot_path

# Function to find and click the edit symbol using template matching
def click_edit_symbol(region, template_path):
    x, y, width, height = region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Template image not found: {template_path}")
    
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    screenshot_gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
    
    result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val >= 0.9:  # Adjust the threshold as needed
        edit_x = max_loc[0] + x
        edit_y = max_loc[1] + y
        pyautogui.click(edit_x+18, edit_y)
        return True
    return False

# Update product price if no discount is detected
def update_product_price(product_coords, region, template_path):
    pyautogui.moveTo(product_coords[0] + 10, product_coords[1])  # Adjust click position 10 pixels to the right
    time.sleep(1)  # Allow time for the hover action to display the edit symbol

    if click_edit_symbol(region, template_path):
        time.sleep(2)
        pyautogui.scroll(-2500)
        time.sleep(2)
        pyautogui.click(percent_sale_coords)
        time.sleep(1)
        pyautogui.typewrite('30')
        time.sleep(1)
        pyautogui.click(apply_sale_coords)
        time.sleep(1)
        pyautogui.click(save_button_coords)
        time.sleep(3)

# Check for keyboard interrupt (Esc key pressed once)
def check_for_exit():
    if keyboard.is_pressed('q'):
        print("Exiting program...")
        exit()

# Coordinates for the initial product (adjust as needed)
initial_product_coords = (306, 534)
percent_sale_coords = (382, 844)
save_button_coords = (1741, 976)
apply_sale_coords = (1182, 683)

# Number of products to update
num_products_per_page = 25
num_scroll = 0

# Coordinates for the region of the first price
first_price_region = (306, 534, 1871-306, 621-534)
edit_symbol_template_path = 'edit.png'  # Path to the edit symbol template image

amount = 0

# Loop through the products on the current page
for i in range(num_products_per_page):
    check_for_exit()  # Check for keyboard interrupt
    
    screenshot_path = screenshot_region(first_price_region, amount)
    amount += 1
    
    # Debugging: Print screenshot path and coordinates
    print(f"Screenshot saved at: {screenshot_path}, Coordinates: {first_price_region}")

    is_discounted = classify_image(screenshot_path)
    
    # Debugging: Print classification result
    print(f"Classification result for product {i+1}: {'Discounted' if is_discounted == 1 else 'Not Discounted'}")

    if is_discounted == 0:
        update_product_price(initial_product_coords, first_price_region, edit_symbol_template_path)
    else:
        print("Skipping product with discount")
        
    pyautogui.scroll(-87)
    time.sleep(2)
    num_scroll += 1

pyautogui.scroll(-2500)
time.sleep(2)
pyautogui.click(1825, 861)  # Next page
