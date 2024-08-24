import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data from the CSV file
c = pd.read_csv("C:\\Users\\rutvi\\OneDrive\\Desktop\\Masters\\Semester 3\\Data science survival skills\\Exercise\\Assignment 7\\classification.csv")

# Print the column names of the DataFrame
print("Column Names:", c.columns)

# Print information about the DataFrame
print(c.info())

# Assuming 'x1', 'x2', and 'label' are the correct column names
X = c[['x1', 'x2']]
y = c['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train an SVM classifier with a polynomial kernel
# You can adjust the degree parameter to control the polynomial degree
classifier = SVC(kernel='poly', degree=3)  # 'poly' for polynomial kernel
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)

# Scatter plot of the data points
sns.scatterplot(x='x1', y='x2', hue='label', data=c, palette='viridis', marker='o', s=75)
plt.title('Scatter Plot of Data with SVM Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')

# Create a meshgrid to plot the decision boundary
xx, yy = np.meshgrid(np.linspace(c['x1'].min(), c['x1'].max(), 100),
                     np.linspace(c['x2'].min(), c['x2'].max(), 100))

# Predict labels for each point in the meshgrid
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

plt.show()
