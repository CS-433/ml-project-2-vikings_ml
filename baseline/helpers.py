""" Several helper functions utilized for the logistic regression and convolution network. 
Most of them are provided by the course staff and ETH. """

import numpy as np
import matplotlib.image as mpimg
import re
from PIL import Image
import os


IMG_WIDTH = 0
IMG_HEIGHT = 0
N_PATCHES_PER_IMAGE = 0
IMG_PATCH_SIZE = 16
PIXEL_DEPTH = 255

def img_float_to_uint8(img):
    """ Converting an image from float to uint8.

    Parameters
    ----------
    img: ndarray
      An array containing the desired image

    Returns
    -------
    rimg: ndarray
      An array containg the desired image, now converted
    """

    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def label_to_img(imgwidth, imgheight, w, h, labels):
    """ Assigning labels to images.

    Parameters
    ----------
    imgwidth: int
      The desired width of the image
    imgheight: int
      The desired height of the image
    w: int
      The width of part of image for each label
    h: int
      The height of part of image for each label
    labels: array
      The labels to assign to the image

    Returns
    -------
    im: ndarray
      An 'image' of the labels
    """

    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j:j + w, i:i + h] = labels[idx]
            idx = idx + 1
    return im


def concatenate_images(img, gt_img):
    """ Concatenating two images, for our use, the predicted image and the groundtruth image for comparison.

    Parameters
    ----------
    img: ndarray
      An array containing the predicted image
    gt_img: ndarray
      An array containing the groundtruth image

    Returns
    -------
    cimg: ndarray
      An array containing the two images concatenated
    """

    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
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


def extract_features(img):
    feat_m = np.mean(img, axis=(0, 1))
    feat_v = np.var(img, axis=(0, 1))
    feat = np.append(feat_m, feat_v)
    return feat


def extract_features_2d(img):
    """ Extracting 2-dimensional features from image, consisting of average gray color and variance. 

    Parameters
    ----------
    img: ndarray
      An array containing the desired image

    Returns
    -------
    feat: ndarray
      An array containing the extracted features
    """

    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat


def extract_img_features(filename, patch_size):
    """ Extracting features for the given image.

    Parameters
    ----------
    filename: str
      The filename for the given image
    patch_size: int
      The patch size for the given image, e.g., 16 for 16x16 patches

    Parameters
    ----------
    X: ndarray
      An array containing the image
    """

    img = mpimg.imread(filename)
    img_patches = mpimg.imread(img, patch_size, patch_size)
    X = np.asarray([extract_features_2d(img_patches[i])
                    for i in range(len(img_patches))])
    return X


def value_to_class(v, foreground_threshold):
    """ Assigning value to class.

    Parameters
    ----------
    v: float
      A float, mainly representing the predicted 'probability' for a patch
    foreground_threshold: float
      The threshold on whether to predict 1 or 0 for the patch

    Returns
    -------
    Either 1 or 0 
    """

    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename, thr):
    """ Reading a single image and outputs the strings that should go into the submission file.

    Parameters
    ------------
    image_filename: string
      The image filename
    thr: float
      The threshold for converting a patch to label

    Yields
    --------
    A formatted prediction string
    """

    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch, thr)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, thr, *image_filenames):
    """ Converting images into a submission file.

    Parameters
    ------------
    submission_filename: string
      the name of the submission file
    thr: float
      The threshold for converting a patch to label
    *image_filenames: list
      list of the image filnames that should be included in the prediction
    """

    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s)
                         for s in mask_to_submission_strings(fn, thr))


def patch_to_label(patch, thr):
    ''' Converting a patch to road if the average pixel value in the patch is larger than the threshold.

    Parameters
    ------------
    patch: ndarray
      An array with predictions or values for a patch
    thr: float
      The threshold for converting a patch to road

    Returns
    --------
    value: int
      1 if the patch is classified as road, 0 otherwise
    '''

    df = np.mean(patch)
    value = 0
    if df > thr:
        value = 1

    return value

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
    color_mask[:, :, 0] = predicted_img * PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

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
    ''' Converting array of labels to an image.
    
    Parameter
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
        
    Returns
    --------
    array_labels: ndarray
        Array of zeros and ones in image format
    '''

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
