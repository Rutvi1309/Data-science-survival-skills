# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:38:31 2023

@author: rutvishah
"""

from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = "C:\\Users\\rutvi\\OneDrive\\Desktop\\docker Hiwi"

img = Image.open(image_path)

# Convert the image to grayscale using Lightness Method
lightness_img = img.convert("L")

# Convert the image to grayscale using Average Method
average_img = img.convert("LA")

# Convert the image to grayscale using Luminosity Method
r, g, b = img.split()
luminosity_img = Image.merge("RGB", (r, g, b))

# Display the original and grayscale images side by side using subplots
plt.figure(figsize=(10, 4))

plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title("Original")

plt.subplot(1, 4, 2)
plt.imshow(lightness_img, cmap="gray")
plt.title("Lightness Method")

plt.subplot(1, 4, 3)
plt.imshow(average_img, cmap="gray")
plt.title("Average Method")

plt.subplot(1, 4, 4)
plt.imshow(luminosity_img)
plt.title("Luminosity Method")

# Hide axes and show the plot
plt.axis("off")
plt.show()
