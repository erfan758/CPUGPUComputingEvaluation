
import requests
from PIL import Image
import numpy as np
import time
from io import BytesIO
import torch
import torchvision.transforms as transforms

# Load an image from a URL for processing
url = "your-url"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# CPU Image Processing
def cpu_image_processing(image):
    start_time = time.time()

    # Perform image processing operations using CPU
    # Example: Convert the image to grayscale
    gray_image = image.convert('L')

    end_time = time.time()
    execution_time = end_time - start_time
    return np.array(gray_image), execution_time

# GPU Image Processing
def gpu_image_processing(image):
    start_time = time.time()

    # Perform image processing operations using GPU
    # Example: Convert the image to grayscale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = transforms.ToTensor()(image).to(device)
    gray_tensor = 0.2989 * image_tensor[0] + 0.5870 * image_tensor[1] + 0.1140 * image_tensor[2]
    gray_image = transforms.ToPILImage()(gray_tensor.cpu())

    end_time = time.time()
    execution_time = end_time - start_time
    return np.array(gray_image), execution_time

# Run CPU Image Processing
cpu_result, cpu_execution_time = cpu_image_processing(image)

# Run GPU Image Processing
gpu_result, gpu_execution_time = gpu_image_processing(image)

# Display the results and execution times
Image.fromarray(cpu_result).show(title="CPU Result")
Image.fromarray(gpu_result).show(title="GPU Result")

print("CPU Execution Time:", cpu_execution_time, "seconds")
print("GPU Execution Time:", gpu_execution_time, "seconds")


