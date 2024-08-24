# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:09:46 2023
@author: rutvishah
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'your_dataset.csv' with the actual file name)
file_path = r"C:\Users\rutvi\OneDrive\Desktop\Semester 3\Data science survival skills\Exercise\Assignment 4\census_income_dataset.csv"
census_data = pd.read_csv(file_path)

# Check the column names
print(census_data.columns)

# a) Age distribution of respondents (Histogram)
plt.figure(figsize=(10, 6))
plt.hist(census_data['AGE'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution of Respondents')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('age_distribution.svg')  # Save as SVG
plt.show()

# b) Relationship status distribution (Bar plot)
plt.figure(figsize=(10, 6))
sns.countplot(x='RELATIONSHIP', data=census_data, palette='viridis')
plt.title('Distribution of Relationship Status')
plt.xlabel('Relationship Status')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.savefig('relationship_distribution.svg')  # Save as SVG
plt.show()

# c) Salary distribution within each educational level (Stacked bar plot)
plt.figure(figsize=(12, 8))
education_salary = census_data.groupby(['EDUCATION', 'SALARY']).size().unstack().fillna(0)
education_salary.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], edgecolor='black')
plt.title('Salary Distribution within Each Educational Level')
plt.xlabel('Educational Level')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Salary', labels=['<=50k', '>50k'])
plt.savefig('salary_distribution.svg')  # Save as SVG
plt.show()
