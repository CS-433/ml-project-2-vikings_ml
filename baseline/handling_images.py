import numpy as np
import os
import matplotlib.image as mpimg
from PIL import Image
import tensorflow as tf

IMG_WIDTH = 0
IMG_HEIGHT = 0
N_PATCHES_PER_IMAGE = 0
IMG_PATCH_SIZE = 16
PIXEL_DEPTH = 255


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

def extract_data(folderpath):
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
    files = os.listdir(folderpath)
    n = len(files)
    imgs = [mpimg.imread(os.path.join(folderpath, files[i])) for i in range(n)]

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
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


def extract_labels(folderpath):
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
    files = os.listdir(folderpath)
    n = len(files)
    for i in range(n):
        img = mpimg.imread(os.path.join(folderpath, files[i]))
        try:
            gt_imgs.append(img[:,:,0])
        except:
            gt_imgs.append(img)

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])
    # Convert to dense 1-hot representation.
    labels = labels.astype(np.float32)
    
    return labels


def label_to_img(imgwidth, imgheight, w, h, labels):
    ''' Convert array of labels to an image 
    
    parameter
    ---------
    imgwidth: int
        Width of image
    imgheight: int
        Height of image
    w: int
        Width of patch
    h: int
        Height of patch
    labels: ndarray
        The labels
        
    returns
    --------
    array_labels: ndarray
        Array of zeros and ones in image format'''

    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:  # bgrd
                l = 0
            else:
                l = 1
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img):
    '''converts image array with floats to uint8
    
    parameters
    -----------
    img: ndarray
        image array
    
    returns
    -------
    rimg: ndarray
        converted array'''

    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

def concatenate_images(img, gt_img):
    '''Concatenate two images
    
    parameters
    -----------
    img: ndarray
        image 1
    gt_img: ndarray
        image 2
    
    returns
    --------
    cimg: ndarray
        concatenated image'''

    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    '''Creates an image with predictions overlayed the groundtruth
    
    parameters
    ------------
    img: ndarray
        groundtruth image
    predicted_img: ndarray
        predicted image
    
    returns
    --------
    new_img: ndarray
        image with overlay'''
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img