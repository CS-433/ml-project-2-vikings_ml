'This file contains the ensemble model that gave the best score on AICrowd with a F1-score of 0.901'
#TODO: fix filepaths
#TODO: Set optimal threshold?

import numpy as np
import pandas as pd

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import segmentation_models as sm
sm.set_framework('tf.keras')
import os
import matplotlib.image as mpimg
from PIL import Image
import cv2
from seg_mod_unet.data_handling import extract_data, extract_data_test, extract_labels
from seg_mod_unet.helpers import patch_to_label, window_predict, img_float_to_uint8, save_predictions
import math
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
# setting the style for plots using seaborn
sns.set()
sns.set_style("white")

# filepath test images
test_images_path = '/content/testing'

# defining backbone for the model
BACKBONE = 'resnet34'
# downloading preprocessing function for the model
preprocess_input = sm.get_preprocessing(BACKBONE)


# model filepaths
model_1 = '/content/drive/MyDrive/ml/m1.h'
model_2 = '/content/drive/MyDrive/ml/m2.h'
model_3 = '/content/drive/MyDrive/ml/m3.h'
model_4 = '/content/drive/MyDrive/ml/m4.h'
model_5 = '/content/drive/MyDrive/ml/m5.h'

# custom objects for the model
custom_objects = {'binary_crossentropy_plus_jaccard_loss':sm.losses.bce_jaccard_loss, 
                      'iou_score': sm.metrics.iou_score, 'f1-score': sm.metrics.FScore()}

# loading models
m1 = load_model(model_1, custom_objects=custom_objects)
m2 = load_model(model_2, custom_objects=custom_objects)
m3 = load_model(model_3, custom_objects=custom_objects)
m4 = load_model(model_4, custom_objects=custom_objects)
m5 = load_model(model_5, custom_objects=custom_objects)
models = [m1, m2, m3, m4, m5]

# loading test images
test_images = extract_data_test('/content/testing/')

#preprocessing test images
test_images = preprocess_input(test_images)

for model in models:

    # generating predictions for the test images
    results = []
    for img in test_images:
        results.append(window_predict(img, model))

    # generating and saving the prediction masks for the testset
    for i in range(1, len(results)+1):
        save_predictions(img, 'test%d'%i)

    # generating prediction csv
    #TODO: Give as input the correct filename to be able to read below
    # generating the prediction file for the test set
    submission_filename = '/content/drive/MyDrive/Pred/model.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = '/content/drive/MyDrive/Pred/test%d.png' % i
        print(image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, foreground_threshold, *image_filenames)





# ensemble model below

# filepaths to the five models' predictions on the test set
X1 = '/content/drive/MyDrive/ml/m1_pred.csv'
X2 = '/content/drive/MyDrive/ml/m2_pred.csv'
X3 = '/content/drive/MyDrive/ml/m3_pred.csv'
X4 = '/content/drive/MyDrive/ml/m4_pred.csv'
X5 = '/content/drive/MyDrive/ml/m5_pred.csv'

# reading the models' prediction into five dataframes

df1 = pd.read_csv(X1)
df1 = df1.set_index(['id'])
df1 = df1.rename({'prediction':'p1'},axis=1)

df2 = pd.read_csv(X2)
df2 = df2.set_index(['id'])
df2 = df2.rename({'prediction':'p2'},axis=1)

df3 = pd.read_csv(X3)
df3 = df3.set_index(['id'])
df3 = df3.rename({'prediction':'p3'},axis=1)

df4 = pd.read_csv(X4)
df4 = df4.set_index(['id'])
df4 = df4.rename({'prediction':'p4'},axis=1)

df5 = pd.read_csv(X5)
df5 = df5.set_index(['id'])
df5 = df5.rename({'prediction':'p5'},axis=1)

# dataframe containing the prediction of all models for each patch
df = pd.concat([df1,df2,df3,df4,df5], axis=1)

# inspecting the dataframe to ensure correct loading
df.head()

# generating predictions, predicting road if all models predict road
df['prediction'] = df.apply(lambda x: 1 if np.sum(x)>4 else 0, axis=1)

# extracting only the prediction column
df = df['prediction']

# writing predctions to csv
df.to_csv('predictions.csv')