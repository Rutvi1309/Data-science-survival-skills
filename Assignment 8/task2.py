import tensorflow as tf
from tensorflow.keras import layers, models

# Define the CNN model using tf.keras.Sequential
model = models.Sequential()

# Add the first convolutional layer with input shape (28, 28, 1) for MNIST images
model.add(layers.Conv2D(8, (3, 3), input_shape=(28, 28), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

# Add the second convolutional layer
model.add(layers.Conv2D(16, (3, 3), input_shape=(14, 14), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Add the third convolutional layer
model.add(layers.Conv2D(32, (3, 3), input_shape=(7, 7), padding='same', activation='relu'))

# Flatten the output for the fully connected layers
model.add(layers.Flatten())

# Add the first fully connected layer
model.add(layers.Dense(128,input shape (1, 1568), activation='relu'))
model.add(Dropout(0.2))
# Add the output layer with 10 units for the 10 classes in MNIST
model.add(layers.Dense(10,input shape (1, 128), activation='softmax'))

# Display the model summary
model.summary()

