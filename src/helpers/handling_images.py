import numpy as np
import os
import matplotlib.image as mpimg
from PIL import Image
import config as cfg

def img_crop(im, w, h):
    ''' (ETH) Extracting patches of width w and height h from an image

    Parameters
    ------------
    im: Image
        The image to extract patches from
    w: int
        The width of the patch
    h: int
        The height of the patch

    Returns
    -------
    list_patches: list
        A list containing patches
    '''
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def extract_data(filename, num_images):
    """ (ETH) Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].

    Parameters
    ----------
    filename: string
        The name of the image file
    num_images: int
        The number of images that should be extracted

    Returns
    -------
    data: ndarray
        A numpy array containting the images
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    cfg.IMG_WIDTH = imgs[0].shape[0]
    cfg.IMG_HEIGHT = imgs[0].shape[1]
    cfg.N_PATCHES_PER_IMAGE = (cfg.IMG_WIDTH/cfg.IMG_PATCH_SIZE)*(cfg.IMG_HEIGHT/cfg.IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], cfg.IMG_PATCH_SIZE, cfg.IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    data = np.asarray(data)
    return data

def value_to_class(v):
    ''' (ETH) Assign labels to a patch v
    
    Parameters
    ----------
    v: ndarray
        The patch
    
    Returns
    --------
    [0,1] if road, [1,0] elsewise
        Labels for the patch'''

    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]


def extract_labels(filename, num_images):
    """ (ETH) Extract the labels into a 1-hot matrix [image index, label index].
    
    Parameters
    ----------
    filename: string
        The name of the image file
    num_images: int
        The number of images
    
    Returns
    --------
    labels: ndarray
        1-hot matrix [image index, label index]
    """
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], cfg.IMG_PATCH_SIZE, cfg.IMG_PATCH_SIZE) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])
    labels = labels.astype(np.float32)
    # Convert to dense 1-hot representation.
    return labels