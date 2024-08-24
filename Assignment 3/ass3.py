import os
from PIL import Image
import matplotlib.pyplot as plt

# Function to load metadata from a .meta file
def load_metadata(meta_path):
    with open(meta_path, 'r') as f:
        metadata = {}
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':')
                metadata[key.strip()] = value.strip()
            else:
                # Handle lines without a colon (':') based on your dataset format
                pass
    return metadata

# Function to load an image and its segmentation mask
def load_image_and_mask(image_path, mask_path):
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    return image, mask

# Function to overlay segmentation mask on the image
def overlay_mask(image, mask):
    # Assuming mask is a binary image (black and white)
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay.paste(mask, mask)
    return Image.alpha_composite(image.convert('RGBA'), overlay)

# Path to the folder containing MiniBAGLS dataset
dataset_folder = "C:\\Users\\rutvi\\OneDrive\\Desktop\\Semester 3\\Data science survival skills\\Exercise\\Assignment 3\\Mini_BAGLS_dataset\\Mini_BAGLS_dataset"

# Get a list of image and mask files
image_files = [f for f in os.listdir(dataset_folder) if f.endswith('.png') and not f.endswith('_seg.png')]
mask_files = [f for f in os.listdir(dataset_folder) if f.endswith('_seg.png')]

# Assuming metadata files have the same name as image files but with a .meta extension
meta_files = [f.replace('.png', '.meta') for f in image_files]

# Check if there are at least 4 images in the dataset
if len(image_files) < 4:
    print("Not enough images in the dataset.")
    print(f"Found {len(image_files)} image files.")
else:
    # Determine the number of rows and columns for subplots
    num_images = len(image_files)
    num_rows = num_images // 2 if num_images % 2 == 0 else (num_images // 2) + 1

    # Plotting the images with segmentation masks overlaid
    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))

    for i in range(num_images):
        image_path = os.path.join(dataset_folder, image_files[i])
        mask_path = os.path.join(dataset_folder, mask_files[i])
        meta_path = os.path.join(dataset_folder, meta_files[i])

        metadata = load_metadata(meta_path)
        image, mask = load_image_and_mask(image_path, mask_path)
        overlayed_image = overlay_mask(image, mask)

        # Plotting
        row = i // 2
        col = i % 2
        axs[row, col].imshow(overlayed_image)
        axs[row, col].set_title(metadata.get('Subject disorder status', 'Unknown'))
        axs[row, col].axis('off')

    plt.show()
