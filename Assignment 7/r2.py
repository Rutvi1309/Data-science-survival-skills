import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

r2 = pd.read_csv("C:\\Users\\rutvi\\OneDrive\\Desktop\\Masters\\Semester 3\\Data science survival skills\\Exercise\\Assignment 7\\regression_2.csv")

# Display the first few rows of the dataframe to understand the structure
print(r2.head())
# Display basic information about the dataframe
print(r2.info())

# Summary statistics
print(r2.describe())

# Check for missing values
print(r2.isnull().sum())


# Assuming 'x1' as the feature and 'x2' as the target
X = r2[['x1']]
y = r2['x2']

# Transform features to cubic
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)

# Fit linear regression with cubic features
model = LinearRegression()
model.fit(X_poly, y)

# Generate predictions
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = poly_features.transform(X_range)
predictions = model.predict(X_range_poly)

# Plot the actual data points and the fitted cubic function
plt.scatter(X, y, label='Actual Data')
plt.plot(X_range, predictions, color='red', label='Cubic Function')
plt.title('Cubic Function for Dataset r2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
