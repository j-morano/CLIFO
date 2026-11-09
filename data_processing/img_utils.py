import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import dilation, disk
from skimage.measure import label
import matplotlib.pyplot as plt


def remove_white_borders(img, show=False):
    img = np.pad(img, 10, constant_values=255)
    bimg = img.copy()
    # Pad image with 255
    bimg = gaussian_filter(bimg, sigma=1)
    bimg = (bimg > 253).astype(int)
    selem = disk(10)
    try:
        bimg = dilation(bimg, selem)
    except IndexError as e:
        print('Error in binary_dilation')
        raise e
    # Get connected components
    label_img = label(bimg, background=0)
    assert isinstance(label_img, np.ndarray)
    # Plot the regions in different colors
    if show:
        plt.imshow(img, cmap='gray')
        plt.imshow(label_img, cmap='nipy_spectral')
        plt.show()
    for v in np.unique(label_img):
        num_pixels = np.sum(label_img == v)
        if num_pixels > 10000 and v != 0:
            img[label_img == v] = 0
    img = img[10:-10, 10:-10]
    if show:
        plt.imshow(img, cmap='gray')
        plt.show()
    return img


