# This file was given by the course with minor changes from us

import os
import numpy as np
import matplotlib.image as mpimg
import re

foreground_threshold = 0.04 # percentage of pixels > 1 required to assign a foreground label to a patch

# Assign a label to a patch
def patch_to_label(patch):
    """ Converting a patch to road if the average pixel value in the patch is larger than the threshold

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
    """

    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """ Reading a single image and outputs the strings that should go into the submission file
    
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
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """ Converting images into a submission file
    
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
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


# Generating the prediction file for the test set
submission_filename = '/content/drive/MyDrive/Pred/m5.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = '/content/drive/MyDrive/Pred/test%d.png' % i
    print(image_filename)
    image_filenames.append(image_filename)
masks_to_submission(submission_filename, *image_filenames)
