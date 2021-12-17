import numpy as np
import matplotlib.image as mpimg
import re

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def concatenate_images(img, gt_img):
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

# Extract 2-dimensional features consisting of average gray color as well as variance


def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract features for a given image


def extract_img_features(filename, patch_size):
    img = mpimg.imread(filename)
    img_patches = mpimg.imread(img, patch_size, patch_size)
    X = np.asarray([extract_features_2d(img_patches[i])
                   for i in range(len(img_patches))])
    return X


def value_to_class(v, foreground_threshold):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def mask_to_submission_strings(image_filename, thr):
    """Reads a single image and outputs the strings that should go into the submission file
    
    parameters
    ------------
    image_filename: string
      The image filename
    thr: float
      The threshold for converting a patch to label
    
    yields
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
    """Converts images into a submission file
    
    parameters
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
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, thr))

def patch_to_label(patch, thr):
  '''
  Converting a patch to road if the average pixel value in the patch is larger than the threshold

  parameters
  ------------
  patch: ndarray
    An array with predictions or values for a patch
  thr: float
    The threshold for converting a patch to road
  
  returns
  --------
  value: int
    1 if the patch is classified as road, 0 otherwise
  '''
  df = np.mean(patch)
  value = 0
  if df > thr:
    value = 1
  
  return value
