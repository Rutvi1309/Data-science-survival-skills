# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:03:57 2023

@author: rutvishah
"""

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "94b980a2",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-14T16:01:51.484205800Z",
     "start_time": "2023-11-14T16:01:49.076002800Z"
    },
    "id": "94b980a2"
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "import numpy as np\n",
    "\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "caf64db9",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-14T16:03:05.522304400Z",
     "start_time": "2023-11-14T16:03:05.445237100Z"
    },
    "id": "caf64db9"
   },
   "outputs": [],
   "source": [
    "BAGLS_PATH = \"Mini_BAGLS_dataset\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "e4cffe71",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-14T16:03:08.691025300Z",
     "start_time": "2023-11-14T16:03:08.588967800Z"
    },
    "id": "e4cffe71",
    "scrolled": True
   },
   "outputs": [],
   "source": [
    "# get all filenames\n",
    "files = os.listdir(BAGLS_PATH)\n",
    "\n",
    "# what is os.listdir() returning?\n",
    "# type ==> list()\n",
    "print(type(files))\n",
    "\n",
    "print(files)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "e592656247939a82",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-14T16:03:11.570608Z",
     "start_time": "2023-11-14T16:03:11.531483200Z"
    }
   },
   "outputs": [],
   "source": [
    "# iterate over the list\n",
    "for f in files:\n",
    "    print(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "7295a9ba0b813f1e",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-14T16:03:12.175503600Z",
     "start_time": "2023-11-14T16:03:12.097930200Z"
    },
    "scrolled": True
   },
   "outputs": [],
   "source": [
    "# get all unique filenames (e.g. each filename with file extension '.meta')\n",
    "filenames = \n",
    "filenames"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "df908d8bbd557988",
   "metadata": {},
   "source": [
    "### Pathlib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "6373c4f7",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-14T16:03:16.785626900Z",
     "start_time": "2023-11-14T16:03:16.695630900Z"
    },
    "id": "6373c4f7",
    "outputId": "1750c04a-9bf0-4273-bdbc-bf6f616bf026"
   },
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "path_pathlib = Path(BAGLS_PATH)\n",
    "path_pathlib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "34348e3e1e1877e6",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-14T16:06:26.115247Z",
     "start_time": "2023-11-14T16:06:26.046038800Z"
    }
   },
   "outputs": [],
   "source": [
    "new_path = path_pathlib / \"64.png\"\n",
    "\n",
    "# Get the parent directory\n",
    "\n",
    "# Get the file name\n",
    "\n",
    "# Separate the number from the file extension\n",
    "\n",
    "\n",
    "# Get the file extension\n",
    "\n",
    "\n",
    "# Get the file name without the extension\n",
    "\n",
    "\n",
    "# Check if the path exists\n",
    "\n",
    "\n",
    "# Check if the path is a file\n",
    "\n",
    "\n",
    "# Check if the path is a directory\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "ef8c3834ca23937f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-14T16:03:19.046113100Z",
     "start_time": "2023-11-14T16:03:18.956486100Z"
    }
   },
   "outputs": [],
   "source": [
    "content = new_path.read_bytes()\n",
    "content"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "dcb3921ae8bfe68c",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-14T16:03:22.898886Z",
     "start_time": "2023-11-14T16:03:22.849095Z"
    },
    "scrolled": True
   },
   "outputs": [],
   "source": [
    "for path in \n",
    "    print(path)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "470ef91470000014",
   "metadata": {},
   "source": [
    "### glob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "4d15e0d1",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-14T16:03:29.652803600Z",
     "start_time": "2023-11-14T16:03:29.537046800Z"
    },
    "id": "4d15e0d1",
    "scrolled": True
   },
   "outputs": [],
   "source": [
    "# another way to combine os.listdir() and the for loop\n",
    "import glob\n",
    "glob_filenames = \n",
    "glob_filenames"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "084eb4eb",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-14T16:03:32.365675700Z",
     "start_time": "2023-11-14T16:03:32.238097Z"
    },
    "id": "084eb4eb",
    "outputId": "7529a4ef-2da4-4ec4-892b-a10442a42647"
   },
   "outputs": [],
   "source": [
    "# iterate over the filenames (with index)\n",
    "for idx, f in enumerate(glob_filenames):\n",
    "    print(idx, f)\n",
    "    break\n",
    "    \n",
    "# out: Index: X, Filename: YYYY"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1714b42fc7c8a291",
   "metadata": {},
   "source": [
    "## Images - General"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "38b2c8a7a6fdef7c",
   "metadata": {
    "ExecuteTime": {
     "start_time": "2023-11-14T16:01:51.512206400Z"
    }
   },
   "outputs": [],
   "source": [
    "################################\n",
    "# Install these packages if you have not done so already\n",
    "################################\n",
    "\n",
    "#!pip install imageio\n",
    "#!pip install scikit-image\n",
    "#!pip install opencv-python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "3a898e7c",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:20:17.395497Z",
     "start_time": "2023-11-15T14:20:17.073366Z"
    },
    "id": "3a898e7c"
   },
   "outputs": [],
   "source": [
    "import imageio.v3 as io\n",
    "from PIL import Image\n",
    "import skimage\n",
    "import cv2\n",
    "\n",
    "## load image with different libraries\n",
    "img_path = \"nice_image.jpg\" # Source: https://www.pexels.com/de-de/foto/picknick-fruchte-liegend-freizeit-18868015/\n",
    "img = io\n",
    "img = Image\n",
    "img = skimage\n",
    "img = cv2\n",
    "\n",
    "# load image as grayscale\n",
    "#img = skimage.io.imread(img_path) #, as_gray=True)\n",
    "#plt.imshow(img) #, cmap='gray')\n",
    "\n",
    "print(img.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "f7e4c91741fd415e",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:20:20.308637200Z",
     "start_time": "2023-11-15T14:20:19.722253200Z"
    }
   },
   "outputs": [],
   "source": [
    "# Histogram of pixel intensities\n",
    "plt.\n",
    "plt.title('Pixel Intensity Histogram')\n",
    "plt.xlabel('Pixel Value')\n",
    "plt.ylabel('Frequency')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "5fe91895d9c58d42",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:20:25.221039800Z",
     "start_time": "2023-11-15T14:20:24.549123400Z"
    }
   },
   "outputs": [],
   "source": [
    "# Basic image manipulation\n",
    "resized_img = \n",
    "plt.imshow(resized_img)\n",
    "plt.show()\n",
    "\n",
    "rotated_img = \n",
    "plt.imshow(rotated_img)\n",
    "plt.show()\n",
    "\n",
    "flipped_img = \n",
    "plt.imshow(flipped_img)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "a3ad3bba203a592c",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:20:33.330629900Z",
     "start_time": "2023-11-15T14:20:32.642316300Z"
    }
   },
   "outputs": [],
   "source": [
    "# Filtering\n",
    "\n",
    "# Apply a Gaussian blur\n",
    "fig, ax = plt.subplots(1, 2, figsize=(10, 5))\n",
    "ax[0].imshow(img)\n",
    "ax[0].set_title(\"Original\")\n",
    "\n",
    "blurred_img = cv2.\n",
    "ax[1].imshow(blurred_img)\n",
    "ax[1].set_title(\"Blurred\")\n",
    "plt.show()\n",
    "\n",
    "# Apply sharpening --> enhance the high-frequency components in an image\n",
    "kernel = \n",
    "img_sharpened = \n",
    "plt.imshow(img_sharpened)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "01163af6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compare the histograms of the original and sharpened image\n",
    "fig, ax = plt.subplots(1, 2, figsize=(10, 5))\n",
    "ax[0].hist(img_gray.ravel(), bins=256, range=[0, 256])\n",
    "ax[0].set_title(\"Original\")\n",
    "\n",
    "ax[1].hist(img_sharpened_gray.ravel(), bins=256, range=[0, 256])\n",
    "ax[1].set_title(\"Sharpened\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "e32699696a1f546f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:20:44.769034300Z",
     "start_time": "2023-11-15T14:20:44.519654Z"
    }
   },
   "outputs": [],
   "source": [
    "# Look at frequency content of the cropped image and img_sharpened\n",
    "from scipy import fftpack\n",
    "\n",
    "def get_magnitude_spectrum(image):\n",
    "    # Compute the 2-dimensional discrete Fourier Transform\n",
    "    \n",
    "\n",
    "    # Shift the zero-frequency component to the center of the spectrum\n",
    "   \n",
    "    \n",
    "    # Compute the magnitude spectrum (logarithmic scale for better visualization)\n",
    "    \n",
    "    \n",
    "    return magnitude_spectrum"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "dc1963b631709e1c",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:20:47.122506100Z",
     "start_time": "2023-11-15T14:20:47.093515900Z"
    }
   },
   "outputs": [],
   "source": [
    "# Convert to grayscale\n",
    "img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #/ 255.\n",
    "img_sharpened_gray = cv2.cvtColor(img_sharpened, cv2.COLOR_RGB2GRAY) #/ 255."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "8618af3a1bf8f29e",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:20:49.095445600Z",
     "start_time": "2023-11-15T14:20:48.454256400Z"
    }
   },
   "outputs": [],
   "source": [
    "mag_spec_original = get_magnitude_spectrum(img_gray)\n",
    "mag_spec_sharpened = get_magnitude_spectrum(img_sharpened_gray)\n",
    "\n",
    "fig, ax = plt.subplots(1, 2, figsize=(10, 5))\n",
    "ax[0].imshow(mag_spec_original)\n",
    "ax[0].set_title(\"Original\")\n",
    "\n",
    "ax[1].imshow(mag_spec_sharpened)\n",
    "ax[1].set_title(\"Sharpened\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "e96d628e",
   "metadata": {
    "id": "e96d628e"
   },
   "outputs": [],
   "source": [
    "# which image loading function to use?\n",
    "import time\n",
    "def test_read_image(imgfile, func):\n",
    "    \n",
    "    return t\n",
    "\n",
    "functions = []\n",
    "\n",
    "times = []\n",
    "for func in functions:\n",
    "    ts = []\n",
    "    for f in path_pathlib.glob(\"*.png\"):\n",
    "        t = \n",
    "        ts.append(t)\n",
    "    times.append(ts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "27ce375c",
   "metadata": {
    "id": "27ce375c"
   },
   "outputs": [],
   "source": [
    "for t, func in zip(times, functions):\n",
    "    print(str(func))\n",
    "    print(\"{:.6f} seconds\".format(np.mean(t)))\n",
    "    print(\"_____\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "21053a9c",
   "metadata": {},
   "source": [
    "## BAGLS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "2ca87821",
   "metadata": {
    "id": "2ca87821",
    "scrolled": True
   },
   "outputs": [],
   "source": [
    "for idx, f in enumerate(path_pathlib.glob(\"*_seg.png\")):\n",
    "    # make sure to only look at the first 5 samples and not all 2000 of them\n",
    "    if idx == 5:\n",
    "        break\n",
    "        \n",
    "    # get basename\n",
    "    basename = \n",
    "    \n",
    "    # load image and segmentation mask\n",
    "    img_path = \n",
    "    seg_path = \n",
    "    img = io.imread(img_path)\n",
    "    seg = io.imread(seg_path)\n",
    "\n",
    "    # visualize image and segmentation mask\n",
    "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))\n",
    "    ax1.axis(\"off\"); ax2.axis(\"off\")\n",
    "\n",
    "    # show both\n",
    "    ax1.imshow(img);ax2.imshow(seg)    \n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "13500c1b",
   "metadata": {
    "id": "13500c1b"
   },
   "outputs": [],
   "source": [
    "ref_image = img\n",
    "plt.imshow(ref_image)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "84d8563c",
   "metadata": {
    "id": "84d8563c"
   },
   "outputs": [],
   "source": [
    "# saving with different formats\n",
    "\n",
    "# tif\n",
    "io.imwrite(\"saved_image.tif\", ref_image)\n",
    "\n",
    "# png\n",
    "io.imwrite(\"saved_image.png\", ref_image)\n",
    "\n",
    "# jpg\n",
    "io.imwrite(\"saved_image.jpg\", ref_image)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "573eeeac",
   "metadata": {
    "id": "573eeeac"
   },
   "outputs": [],
   "source": [
    "# check out memory footprints\n",
    "import os\n",
    "img_tif_size = os.path.getsize(\"saved_image.tif\")\n",
    "img_png_size = os.path.getsize(\"saved_image.png\")\n",
    "img_jpg_size = os.path.getsize(\"saved_image.jpg\")\n",
    "\n",
    "plt.bar([0, 1, 2], [img_tif_size, img_png_size, img_jpg_size])\n",
    "plt.xticks([0, 1, 2], [\".tif\", \".png\", \".jpg\"])\n",
    "plt.ylabel(\"File size [bytes]\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b8fabdee",
   "metadata": {
    "id": "b8fabdee"
   },
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "77b07ac9",
   "metadata": {
    "id": "77b07ac9"
   },
   "outputs": [],
   "source": [
    "# load images\n",
    "img_tif = io.imread(\"saved_image.tif\")\n",
    "img_png = io.imread(\"saved_image.png\")\n",
    "img_jpg = io.imread(\"saved_image.jpg\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "12675b5f",
   "metadata": {
    "id": "12675b5f",
    "outputId": "6e7c302a-61e7-45b5-cd16-3d18967822ab"
   },
   "outputs": [],
   "source": [
    "# compare them to each other\n",
    "np.allclose(ref_image, img_tif)\n",
    "np.allclose(ref_image, img_png)\n",
    "np.allclose(ref_image, img_jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "22c98c7e",
   "metadata": {
    "id": "22c98c7e"
   },
   "source": [
    "### Calculate [PSNR](https://www.ni.com/de-de/innovations/white-papers/11/peak-signal-to-noise-ratio-as-an-image-quality-metric.html) between images"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "id": "1d292626",
   "metadata": {},
   "source": [
    "![image.png](attachment:image.png)"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "id": "5ebb2e2d",
   "metadata": {},
   "source": [
    "![image.png](attachment:image.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "91842aee",
   "metadata": {
    "id": "91842aee"
   },
   "outputs": [],
   "source": [
    "def psnr(img1, img2):\n",
    "    #mean squared error:\n",
    "    mse = \n",
    "    \n",
    "    return "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "edadb763",
   "metadata": {
    "id": "edadb763",
    "outputId": "4ade2c85-9b7f-4150-a9b2-8c280d4ea11d"
   },
   "outputs": [],
   "source": [
    "print(psnr(ref_image, img_tif))\n",
    "print(psnr(ref_image, img_png))\n",
    "print(psnr(ref_image, img_jpg))\n",
    "\n",
    "print(psnr(ref_image, np.random.randint(0, 255, ref_image.shape)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "3b2a111f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:29:51.929406200Z",
     "start_time": "2023-11-15T14:29:51.784352800Z"
    },
    "id": "3b2a111f",
    "outputId": "199929c6-5e6f-4eed-f11c-442978784655"
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# get all meta data information \n",
    "all_meta_data = []\n",
    "\n",
    "for f in glob_filenames:\n",
    "    # open .meta file\n",
    "    meta_file = open(f)\n",
    "    meta_data = json.load(meta_file)\n",
    "    all_meta_data.append(meta_data)\n",
    "    \n",
    "df = pd.DataFrame(all_meta_data)\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "b663ab53",
   "metadata": {
    "id": "b663ab53"
   },
   "outputs": [],
   "source": [
    "df.to_csv(\"metadata.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "69937030",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:25:37.296726400Z",
     "start_time": "2023-11-15T14:25:36.903527300Z"
    },
    "id": "69937030"
   },
   "outputs": [],
   "source": [
    "vid_path = \"glottis_video/glottis_video.mp4\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "1d5cff41",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:25:38.162901500Z",
     "start_time": "2023-11-15T14:25:38.130891900Z"
    },
    "id": "1d5cff41",
    "outputId": "d40c3758-e9b1-41d5-8ebe-96e5d81bfdf5"
   },
   "outputs": [],
   "source": [
    "from IPython.display import Video\n",
    "Video(vid_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "a588f15e5b3bac01",
   "metadata": {},
   "outputs": [],
   "source": [
    "# If needed, install scikit-video\n",
    "#!pip install scikit-video"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "0ebb79b1",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:26:32.556239100Z",
     "start_time": "2023-11-15T14:26:31.695036600Z"
    },
    "id": "0ebb79b1",
    "outputId": "64ec3ef6-bb92-4397-84de-843d574b53ae"
   },
   "outputs": [],
   "source": [
    "# load video using scikit-video\n",
    "\n",
    "import skvideo.io  \n",
    "glottis_video = skvideo.io.vread(vid_path)  \n",
    "glottis_video.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "f95486ae5d2d99a0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# If needed\n",
    "# !pip install imageio-ffmpeg"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "73c286df",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:27:02.120840600Z",
     "start_time": "2023-11-15T14:27:01.548265Z"
    },
    "id": "73c286df",
    "outputId": "edfea8fd-4d29-4ba9-c4ed-5c888e8aa5e1"
   },
   "outputs": [],
   "source": [
    "# load video using imageio\n",
    "\n",
    "import imageio\n",
    "glottis_video = imageio.mimread(vid_path)\n",
    "np.asarray(glottis_video).shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "f6f33d32",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:27:30.575131Z",
     "start_time": "2023-11-15T14:27:30.273962Z"
    },
    "colab": {
     "referenced_widgets": [
      "79d927d76a724e80ad9c70baee826452"
     ]
    },
    "id": "f6f33d32",
    "outputId": "447eb85c-db51-4a44-b623-9fcec0dc7135"
   },
   "outputs": [],
   "source": [
    "import ipywidgets as widgets\n",
    "\n",
    "alpha_slider = widgets.FloatSlider(\n",
    "    value=1.,\n",
    "    min=0,\n",
    "    max=1.0,\n",
    "    step=0.01,\n",
    "    description='Alpha',\n",
    ")\n",
    "\n",
    "@widgets.interact(n=(0, len(glottis_video)-1))\n",
    "def f(n=5, alpha=alpha_slider):\n",
    "    alpha = alpha_slider.value\n",
    "    plt.imshow(glottis_video[n], alpha=alpha)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "b28ff830",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:27:45.396434500Z",
     "start_time": "2023-11-15T14:27:45.106377400Z"
    },
    "id": "b28ff830",
    "outputId": "350b8e68-194f-4aa6-f9e3-8cc6760b731d"
   },
   "outputs": [],
   "source": [
    "# load video frame by frame by creating an iterable reader object\n",
    "\n",
    "vid_reader = imageio.get_reader(vid_path,  'ffmpeg')\n",
    "\n",
    "for frame in vid_reader.iter_data():\n",
    "    plt.imshow(frame)\n",
    "    break"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "f89ab1b8",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:27:48.279051600Z",
     "start_time": "2023-11-15T14:27:48.184052400Z"
    },
    "id": "f89ab1b8",
    "outputId": "99c6139e-e6ed-4b86-e744-65c0772a0155"
   },
   "outputs": [],
   "source": [
    "# JSON files\n",
    "metadata = vid_reader.get_meta_data()\n",
    "metadata"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "722360b2",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:27:50.816876700Z",
     "start_time": "2023-11-15T14:27:50.792572900Z"
    },
    "id": "722360b2"
   },
   "outputs": [],
   "source": [
    "import json\n",
    "\n",
    "with open('video_metadata_example.json', 'w') as f:\n",
    "    json.dump(metadata, f, indent=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count":None,
   "id": "13a20b77",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:27:52.276440200Z",
     "start_time": "2023-11-15T14:27:52.223727800Z"
    },
    "id": "13a20b77",
    "outputId": "467727d8-50b5-4777-8079-a4f49ba45b41"
   },
   "outputs": [],
   "source": [
    "with open('video_metadata_example.json', 'r') as f:\n",
    "    data = json.load(f)\n",
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "2b0f700b",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:28:52.573990900Z",
     "start_time": "2023-11-15T14:28:51.201905100Z"
    },
    "id": "2b0f700b",
    "outputId": "d4056beb-c7bf-479d-888b-c5b2a21f013e"
   },
   "outputs": [],
   "source": [
    "# save videos lossless\n",
    "\n",
    "imageio.mimwrite(\"saved_video.mp4\", \n",
    "                 glottis_video,   # video \n",
    "                 fps=10,          # frames per second\n",
    "                 codec='libx264rgb',   # use the right codec\n",
    "                 pixelformat='rgb24',   # and pixel format\n",
    "                 output_params=['-crf', '0',      # Ensure setting crf to 0\n",
    "                                '-preset', 'veryslow']) # Maximum compression: veryslow, \n",
    "                                                         # maximum speed: ultrafast\n",
    "    \n",
    "# check out file size depending on 'veryslow' or 'ultrafast'\n",
    "os.path.getsize(\"saved_video.mp4\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "53b00d94",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:28:05.896412900Z",
     "start_time": "2023-11-15T14:28:05.762194900Z"
    },
    "id": "53b00d94",
    "outputId": "865d2fd9-c748-4dde-9437-2b29268bf6bf"
   },
   "outputs": [],
   "source": [
    "# save videos lossless\n",
    "\n",
    "imageio.mimwrite(\"saved_video.mp4\", \n",
    "                 glottis_video,   # video \n",
    "                 fps=10,          # frames per second\n",
    "                 codec='libx264rgb',   # use the right codec\n",
    "                 pixelformat='rgb24',   # and pixel format\n",
    "                 output_params=['-crf', '0',      # Ensure setting crf to 0\n",
    "                                '-preset', 'ultrafast']) # Maximum compression: veryslow, \n",
    "                                                         # maximum speed: ultrafast\n",
    "    \n",
    "# check out file size depending on 'veryslow' or 'ultrafast'\n",
    "os.path.getsize(\"saved_video.mp4\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "b41cfceb",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:28:54.867109Z",
     "start_time": "2023-11-15T14:28:54.319181Z"
    },
    "id": "b41cfceb",
    "outputId": "08da600c-bfe7-4d00-dd0a-6fa8c1caba17"
   },
   "outputs": [],
   "source": [
    "# compare original loaded video and the saved video\n",
    "\n",
    "saved_vid = imageio.mimread(\"saved_video.mp4\")\n",
    "np.allclose(glottis_video, saved_vid)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "cef41089231cfd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# If needed\n",
    "#!pip install flammkuchen"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "cd362d26",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:30:01.888506Z",
     "start_time": "2023-11-15T14:30:01.648655200Z"
    },
    "id": "cd362d26"
   },
   "outputs": [],
   "source": [
    "# Flammkuchen\n",
    "import flammkuchen as fl\n",
    "\n",
    "d = {\n",
    "    'tabular': pd.DataFrame(np.random.random((20,40))), # random tabular data\n",
    "     'videos': np.random.randint(0, 256, (30, 256, 256, 3)).astype(np.uint8), # random video\n",
    "     'json': dict(name='John Doe', age=32, gender=\"d\") # random meta data\n",
    "}\n",
    "\n",
    "fl.save(\"my_hdf5_file.h5\", d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "4089c8ec",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:30:03.902278Z",
     "start_time": "2023-11-15T14:30:03.863601200Z"
    },
    "id": "4089c8ec",
    "outputId": "f7144593-3979-47b0-cbdb-2410a77366a6"
   },
   "outputs": [],
   "source": [
    "fl.meta(\"my_hdf5_file.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "1c98b628",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:30:06.591845200Z",
     "start_time": "2023-11-15T14:30:06.549746200Z"
    },
    "id": "1c98b628",
    "outputId": "66d843c6-33a5-443e-992e-b52dd9ddea9c"
   },
   "outputs": [],
   "source": [
    "random_video = fl.load(\"my_hdf5_file.h5\", \"/videos\")\n",
    "random_video.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "27a0c423",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-15T14:30:09.096127300Z",
     "start_time": "2023-11-15T14:30:08.987499100Z"
    },
    "id": "27a0c423",
    "outputId": "1160ca8a-1fc4-4602-dd89-aee13cdaa729"
   },
   "outputs": [],
   "source": [
    "# And only a portion of the video, e.g. only the red channel of the first 5 frames\n",
    "r_first_5_frames = fl.load(\"my_hdf5_file.h5\", \"/videos\", sel=fl.aslice[:5, ..., 0])\n",
    "r_first_5_frames.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count":None,
   "id": "5820da64",
   "metadata": {
    "id": "5820da64"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [
    "b8fabdee",
    "22c98c7e"
   ],
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
