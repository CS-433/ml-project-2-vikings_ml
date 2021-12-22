""" Several helper functions utilized in final_model.ipynb """

import numpy as np
import cv2
import re
import matplotlib.image as mpimg
from sklearn.metrics import f1_score

def patch_to_label(patch, thr):
  ''' Converting a patch to road if the average pixel value in the patch is larger than the threshold

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

def window_predict(img, model):
    ''' Predicting segmenation on an image using the window method, i.e. predicting on 256x256 crops of the image.
    
    Parameters
    ------------
    img: ndarray
        An image that should be segmented
    model: Keras model
      The model that should make predictions
    
    Returns
    ------------
    pred: ndarray
        The predicted image
    '''

    # Cropping the image into images of size 256x256
    img1 = img[0:256,0:256,:]
    img2 = img[0:256,256:512,:]
    img3 = img[0:256,352:608,:]
    img4 = img[256:512,0:256,:]
    img5 = img[256:512,256:512,:]
    img6 = img[256:512,352:608,:]
    img7 = img[352:608,0:256,:]
    img8 = img[352:608,256:512,:]
    img9 = img[352:608,352:608,:]

    # Predicting on each of the cropped images
    pred_1 = model.predict(np.expand_dims(img1, axis=0))[0]
    pred_2 = model.predict(np.expand_dims(img2, axis=0))[0]
    pred_3 = model.predict(np.expand_dims(img3, axis=0))[0]
    pred_4 = model.predict(np.expand_dims(img4, axis=0))[0]
    pred_5 = model.predict(np.expand_dims(img5, axis=0))[0]
    pred_6 = model.predict(np.expand_dims(img6, axis=0))[0]
    pred_7 = model.predict(np.expand_dims(img7, axis=0))[0]
    pred_8 = model.predict(np.expand_dims(img8, axis=0))[0]
    pred_9 = model.predict(np.expand_dims(img9, axis=0))[0]

    # Cropping the images which are to the right in the original image
    pred_3 = pred_3[:,160:256,:]
    pred_6 = pred_6[:,160:256,:]
    pred_9 = pred_9[:,160:256,:]

    # Stacking images horizontally into three parts, top, middle, and bottom of the original image
    top = np.hstack([np.hstack([pred_1,pred_2]),pred_3])
    middle = np.hstack([np.hstack([pred_4,pred_5]),pred_6])
    bottom = np.hstack([np.hstack([pred_7,pred_8]),pred_9])

    # Cropping the bottom
    bottom = bottom[160:256,:,:]

    # Stacking top, middle, and bottom to create finished prediction
    pred = np.vstack([np.vstack([top, middle]), bottom])
    
    return pred


def save_predictions(img, name):
  ''' Saving an image.
  
  pParameters
  -----------
  img: ndarray
    The image that should be saved
  name: string
    The filename for the image
  '''

  # Converting the image from one to three channels
  w = img.shape[0]
  h = img.shape[1]
  gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
  gt_img8 = img_float_to_uint8(img)          
  gt_img_3c[:, :, 0] = gt_img8[:,:,0]
  gt_img_3c[:, :, 1] = gt_img8[:,:,0]
  gt_img_3c[:, :, 2] = gt_img8[:,:,0]

  # Saving the image
  cv2.imwrite('predictions/%s.png'%(name), gt_img_3c)

def mask_to_submission_strings(image_filename, thr):
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
            label = patch_to_label(patch, thr)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, thr, *image_filenames):
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
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, thr))

def img_float_to_uint8(img):
    """ Converting image array with floats to uint8
    
    Parameters
    -----------
    img: ndarray
        image array
    
    Returns
    -------
    rimg: ndarray
        converted array
    """

    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def test_threshold(preds, gts, min, max):
  """ Finding the optimal threshold for labeling a patch as road. Searches with stepsize 0.01

  Parameters
  -----------
  preds: ndarray
    Pixelwise label predictions of an iamge
  gts:
    Pixelwise labels for the original image
  min:
    100 times the value to start searching for an optimum
  max:
    100 times the value for stopping searching for an optimum
  """

  # Defining a list of potential thresholds
  thresholds = [0.01*i for i in range(min, max)]
  
  # List for saving f1-scores
  f1s = []
  
  # Saving highest f1-score 
  highest = 0
  foreground_threshold = 0
  patch_size = 16
  
  # Iterating through each threshold and calculating F1-score and accuracy
  for thr in thresholds:

    # Converting pixelwise predictions to patchwise predictions
    y_pred_flattened = []
    for im in preds:
      for j in range(0, im.shape[1], patch_size):
            for i in range(0, im.shape[0], patch_size):
                patch = im[i:i + patch_size, j:j + patch_size]
                label = patch_to_label(patch, thr)
                y_pred_flattened.append(label)
    y_pred_flattened = np.array(y_pred_flattened)

    # Converting mask to patchwise values
    y_val_flattened = []
    for im in gts:
      for j in range(0, im.shape[1], patch_size):
            for i in range(0, im.shape[0], patch_size):
                patch = im[i:i + patch_size, j:j + patch_size]
                label = patch_to_label(patch, thr)
                y_val_flattened.append(label)

    # Calculating and storing f1-score and accuracy
    f1 = f1_score(y_val_flattened, y_pred_flattened)
    f1s.append(f1)

    # Setting foreground_threshold for future predcitions to thr if thr gives the best f1-score
    if f1>highest:
      foreground_threshold=thr
      highest = f1

  print("The best threshold is: %.2f and achieves a F1-score of : %.4f"%(foreground_threshold, highest))
  return foreground_threshold