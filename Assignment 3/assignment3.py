import os
from PIL import Image
import matplotlib.pyplot as plt

# Function to load an image and its segmentation mask
def load_image_and_mask(image_path, mask_path):
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    return image, mask

# Function to overlay segmentation mask on the image
def overlay_mask(image, mask):
    # Change the RGBA value here (e.g., semi-transparent red overlay)
    overlay = Image.new('RGBA', image.size, (0, 0, 100000000, 108))
    overlay.paste(mask, mask)
    return Image.alpha_composite(image.convert('RGBA'), overlay)

# Path to the folder containing MiniBAGLS dataset
dataset_folder = "C:\\Users\\rutvi\\OneDrive\\Desktop\\Semester 3\\Data science survival skills\\Exercise\\Assignment 3\\Mini_BAGLS_dataset\\Mini_BAGLS_dataset"

# Get a list of image and mask files
image_files = [f for f in os.listdir(dataset_folder) if f.endswith('.png') and not f.endswith('_seg.png')][:4]
mask_files = [f.replace('.png', '_seg.png') for f in image_files]

# Check if there are at least 4 images in the dataset
if len(image_files) < 4:
    print("Not enough images in the dataset.")
    print(f"Found {len(image_files)} image files.")
else:
    # Plotting the first four images with segmentation masks overlaid
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i in range(4):
        image_path = os.path.join(dataset_folder, image_files[i])
        mask_path = os.path.join(dataset_folder, mask_files[i])

        # Extract subject disorder status from the file name
        subject_status = "Healthy" if "healthy" in image_files[i].lower() else "Unhealthy"

        image, mask = load_image_and_mask(image_path, mask_path)
        overlayed_image = overlay_mask(image, mask)

        # Plotting
        row = i // 2
        col = i % 2
        axs[row, col].imshow(overlayed_image)
        axs[row, col].set_title("Healthy")
        axs[row, col].axis('off')

    plt.show()
