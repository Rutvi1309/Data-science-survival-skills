import numpy as np
from skimage import data, color
from skimage.transform import resize
from line_profiler import LineProfiler
from joblib import Parallel, delayed

# Load data
imgs = np.uint8(data.lfw_subset() * 255)

def res_skimage(imgs):
    new_size = (imgs[1].shape[0] // 2, imgs[1].shape[1] // 2)

    def resize_image(im):
        return resize(im, new_size, anti_aliasing=True)

    res_im = Parallel(n_jobs=-1)(delayed(resize_image)(im) for im in imgs)

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
