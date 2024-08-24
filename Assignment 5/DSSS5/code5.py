import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Update the file path with a raw string (r) and without double quotes
file_path = r'C:\Users\rutvi\OneDrive\Desktop\Semester 3\Data science survival skills\Exercise\Assignment 5\winequality-red.csv'

# Load the dataset
wine_data = pd.read_csv(file_path)

# Separate features and labels
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Function to plot the data
def plot_data(X_2d, title, color_labels, ax):
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=color_labels, cmap='viridis', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(*scatter.legend_elements(), title='Quality')

# Plot using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
plot_data(X_pca, 'PCA', y, ax1)

# Plot using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
fig, ax2 = plt.subplots(1, 1, figsize=(12, 6))
plot_data(X_tsne, 't-SNE', y, ax2)

# Plot using UMAP
umap_model = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
X_umap = umap_model.fit_transform(X)
fig, ax3 = plt.subplots(1, 1, figsize=(12, 6))
plot_data(X_umap, 'UMAP', y, ax3)

plt.show()
