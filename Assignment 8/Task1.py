import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Plot a random sample
random_index = np.random.randint(0, len(train_images))
random_image = train_images[random_index]
label = train_labels[random_index]

# Display the image and label
plt.imshow(random_image, cmap='gray')
plt.title(f"Label: {label}")
plt.show()
