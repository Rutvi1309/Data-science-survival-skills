import numpy as np
from skimage import data, color
from skimage.transform import resize
from line_profiler import LineProfiler

# Load data
imgs = np.uint8(data.lfw_subset() * 255)

def res_skimage(imgs):
    new_size = (imgs[1].shape[0] // 2, imgs[1].shape[1] // 2)
    res_im = []

    for im in imgs:
        image_resized = resize(im, new_size, anti_aliasing=True)
        res_im.append(image_resized)

    return np.asarray(res_im)

# Create a profile object
profiler = LineProfiler()

# Add the function to be profiled to the profiler
profiler.add_function(resize)
profiler.add_function(res_skimage)

# Run the profiler on the function
profiler.run('res_skimage(imgs)')

# Print the profiling results
profiler.print_stats()
