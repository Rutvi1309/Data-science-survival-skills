import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load the RGB image
image_path = "C:\\Users\\rutvi\\OneDrive\\Desktop\\Semester 3\\Data science survival skills\\Exercise\\Assignment 3\\leaves.jpg"
rgb_image = Image.open(image_path)

# Convert the RGB image to a NumPy array
rgb_array = np.array(rgb_image)

# Define the luminosity method function
def luminosity_method(rgb):
    return 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]

# Define the lightness method function
def lightness_method(rgb):
    return (np.min(rgb, axis=2) + np.max(rgb, axis=2)) / 2

# Define the average method function
def average_method(rgb):
    return np.mean(rgb, axis=2)

# Convert the RGB array to grayscale using different methods
gray_luminosity = luminosity_method(rgb_array)
gray_lightness = lightness_method(rgb_array)
gray_average = average_method(rgb_array)

# Display the original RGB image and the grayscale images side by side
plt.figure(figsize=(15, 4))

# Original RGB image
plt.subplot(1, 4, 1)
plt.imshow(rgb_array)
plt.title('Original RGB')

# Grayscale using luminosity method
plt.subplot(1, 4, 2)
plt.imshow(gray_luminosity, cmap='gray')
plt.title('Grayscale (Luminosity)')

# Grayscale using lightness method
plt.subplot(1, 4, 3)
plt.imshow(gray_lightness, cmap='gray')
plt.title('Grayscale (Lightness)')

# Grayscale using average method
plt.subplot(1, 4, 4)
plt.imshow(gray_average, cmap='gray')
plt.title('Grayscale (Average)')

# Show the plots
plt.tight_layout()
plt.show()
