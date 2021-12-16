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

def window_predict(img):
    '''
    Predicts segmenation on an image using the window method, i.e. predicting on 256x256 crops of the image.
    
    parameters
    ------------
    img: ndarray
        An image that should be segmented
    
    returns
    ------------
    pred: ndarray
        The predicted image
    '''
    # cropping the images into images of size 256x256
    img1 = img[0:256,0:256,:]
    img2 = img[0:256,256:512,:]
    img3 = img[0:256,352:608,:]
    img4 = img[256:512,0:256,:]
    img5 = img[256:512,256:512,:]
    img6 = img[256:512,352:608,:]
    img7 = img[352:608,0:256,:]
    img8 = img[352:608,256:512,:]
    img9 = img[352:608,352:608,:]

    # predictiong on each of the cropped images
    pred_1 = model.predict(np.expand_dims(img1, axis=0))[0]
    pred_2 = model.predict(np.expand_dims(img2, axis=0))[0]
    pred_3 = model.predict(np.expand_dims(img3, axis=0))[0]
    pred_4 = model.predict(np.expand_dims(img4, axis=0))[0]
    pred_5 = model.predict(np.expand_dims(img5, axis=0))[0]
    pred_6 = model.predict(np.expand_dims(img6, axis=0))[0]
    pred_7 = model.predict(np.expand_dims(img7, axis=0))[0]
    pred_8 = model.predict(np.expand_dims(img8, axis=0))[0]
    pred_9 = model.predict(np.expand_dims(img9, axis=0))[0]

    #cropping the images which are to the right in the original image
    pred_3 = pred_3[:,160:256,:]
    pred_6 = pred_6[:,160:256,:]
    pred_9 = pred_9[:,160:256,:]

    # stacking images horizontally into three parts, top, middle, and bottom of the original image
    top = np.hstack([np.hstack([pred_1,pred_2]),pred_3])
    middle = np.hstack([np.hstack([pred_4,pred_5]),pred_6])
    bottom = np.hstack([np.hstack([pred_7,pred_8]),pred_9])

    # cropping the bottom
    bottom = bottom[160:256,:,:]

    # stacking top, middle, and bottom to create finished prediction
    pred = np.vstack([np.vstack([top, middle]), bottom])
    
    return pred


def save_predictions(img, name):
  '''
  Saves an image
  
  parameters
  -----------
  img: ndarray
    The image that should be saved
  name: string
    The filename for the image
  '''
  # converting the image from one to three channels
  w = img.shape[0]
  h = img.shape[1]
  gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
  gt_img8 = img_float_to_uint8(img)          
  gt_img_3c[:, :, 0] = gt_img8[:,:,0]
  gt_img_3c[:, :, 1] = gt_img8[:,:,0]
  gt_img_3c[:, :, 2] = gt_img8[:,:,0]

  # saving the image
  cv2.imwrite('/content/drive/MyDrive/ml/%s.png'%(name), gt_img_3c)

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
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg