import numpy as np
import matplotlib.pyplot as plt
import flammkuchen as fl

def calculate_iou(rect1, rect2):
    def to_numeric(rect):
        try:
            return tuple(map(int, rect))
        except ValueError:
            # Handle non-numeric values, for example, by skipping the rectangle
            return None

    rect1_numeric = to_numeric(rect1)
    rect2_numeric = to_numeric(rect2)

    if rect1_numeric is None or rect2_numeric is None:
        return 0  # Return 0 for non-numeric rectangles

    x1, y1, w1, h1 = rect1_numeric
    x2, y2, w2, h2 = rect2_numeric

    intersection_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersection_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    intersection_area = intersection_x * intersection_y
    union_area = w1 * h1 + w2 * h2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

# Provide the path to your HDF5 file
file_path = r'C:\Users\rutvi\OneDrive\Desktop\Masters\Semester 3\Data science survival skills\Exercise\Assignment 6\rectanglesdsss.sec'

# Load data from the HDF5 file using flammkuchen
data = fl.load(file_path)

# Access the data using dictionary keys
ground_truth_rectangles = np.array(data['ground_truth'])
predicted_rectangles = np.array(data['predicted'])

# Calculate IoU scores
iou_scores = [calculate_iou(gt, pred) for gt, pred in zip(ground_truth_rectangles, predicted_rectangles)]

# Plot histogram
plt.hist(iou_scores, bins=20, color='blue', edgecolor='black')
plt.title('Distribution of IoU Scores ')
plt.xlabel('IoU ')
plt.ylabel('Number of Rectangles')
plt.show()
