'This file contains the ensemble model that gave the best score on AICrowd with a F1-score of 0.901'
#TODO: Set optimal threshold?
#TODO: Test code
#TODO: Add my final datafolder, the one without splits

# imports
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import segmentation_models as sm
sm.set_framework('tf.keras')
from seg_mod_unet.data_handling import extract_data, extract_data_test, extract_labels
from seg_mod_unet.helpers import window_predict, save_predictions, masks_to_submission
from pathlib import Path
import os

# data paths
data_path = os.path.join(str(Path.cwd()), 'data')
train_path = os.path.join(data_path, 'training')
train_data_path = os.path.join(train_path, 'images')
train_labels_path = os.path.join(train_path, 'groundtruth')
test_images_path = os.path.join(data_path, 'testing')

#TODO: How to store models?
# model paths
model_folder = os.path.join(str(Path.cwd()), 'models')
m1_path = os.path.join(model_folder, 'm1.h')
m2_path = os.path.join(model_folder, 'm2.h')
m3_path = os.path.join(model_folder, 'm3.h')
m4_path = os.path.join(model_folder, 'm4.h')
m5_path = os.path.join(model_folder, 'm5.h')

# filepaths to the five models' predictions on the test set
X1 = 'predictions/m1_pred.csv'
X2 = 'predictions/m2_pred.csv'
X3 = 'predictions/m3_pred.csv'
X4 = 'predictions/m4_pred.csv'
X5 = 'predictions/m5_pred.csv'

# defining if model should be trained or not
TRAIN = False

# defining backbone for the model
BACKBONE = 'resnet34'

# downloading preprocessing function for the model
preprocess_input = sm.get_preprocessing(BACKBONE)

#defining loss, regularizer and optimizer for the model
optimizer = 'Adam'
kernel_regularizer = keras.regularizers.l2(1)
loss = sm.losses.bce_jaccard_loss

# defining number of epochs and batch size
num_epcohs = 50
batch_size = 32

# custom objects for the model
custom_objects = {'binary_crossentropy_plus_jaccard_loss':loss, 
                      'iou_score': sm.metrics.iou_score, 'f1-score': sm.metrics.FScore()}
                      
# defining threhshold for attributing patch as road
foreground_threshold = 0.04


def main():
    if TRAIN:
        #TODO: Test if this code is memory robust

        # Extracting the data and masks
        x = extract_data(train_data_path)
        y = extract_labels(train_labels_path)

        # training 5 seperate models
        for i in range(0, 5):
            x_train = x
            y_train = y
            # Splitting the dataset into two, one training set and one validation set
            #TODO: Test this code
            x_val = x_train[i*340:(i+1)*340]
            y_val = y_train[i*340:(i+1)*340]
            x_train = np.delete(x_train, np.s_[i*340:(i+1)*340])
            y_train = np.delete(y_train, np.s_[i*340:(i+1)*340])

            # preprocessing input
            x_train = preprocess_input(x_train)
            x_val = preprocess_input(x_val)

            # defining model, using 'imagenet' as weights to converge faster
            model = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape=(256, 256, 3))

            # adding  L2 kernel regularizer
            sm.utils.set_regularization(model, kernel_regularizer=kernel_regularizer)

            # compiling the model using Adam optimizer and Binary Cross Entropy with Jaccard loss
            model.compile(
                optimizer,
                loss=loss,
                metrics=[sm.metrics.iou_score, sm.metrics.FScore(),'accuracy'],
            )

            # saving the model thats scores best on the validation data
            callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(model_folder, "m%d.h5") % (i+1), save_best_only=True)]

            # training the model for 50 epochs with batch size = 32
            history = model.fit(x=x_train, y=y_train,
            epochs=num_epochs, batch_size=batch_size,
            callbacks=callbacks,
            validation_data=(x_val,y_val)
            )



    # loading models
    m1 = load_model(m1_path, custom_objects=custom_objects)
    m2 = load_model(m2_path, custom_objects=custom_objects)
    m3 = load_model(m3_path, custom_objects=custom_objects)
    m4 = load_model(m4_path, custom_objects=custom_objects)
    m5 = load_model(m5_path, custom_objects=custom_objects)

    models = [m1, m2, m3, m4, m5]

    # loading test images
    test_images = extract_data_test(test_images_path)

    #preprocessing test images
    test_images = preprocess_input(test_images)

    for i in range(len(models)):
        # generating predictions for the test images
        results = []
        for img in test_images:
            results.append(window_predict(img, models[i]))

        # generating and saving the prediction masks for the testset
        for i in range(1, len(results)+1):
            save_predictions(img, 'test%d'%i)

        # generating the prediction file for the test set
        submission_filename = 'predictions/m%d_pred.csv' % (i+1)
        image_filenames = []
        for j in range(1, 51):
            image_filename = 'predictions/test%d.png' % j
            image_filenames.append(image_filename)
        masks_to_submission(submission_filename, foreground_threshold, *image_filenames)

    " Making ensemble model predictions "

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
    df.to_csv('predctions/ensemble.csv')


if __name__ == "__main__":
    main()
