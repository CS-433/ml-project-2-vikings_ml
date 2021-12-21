""" This file contains the ensemble model that gave the best score on AICrowd with a F1-score of 0.901. """
# TODO: Test code
# TODO: Test with fresh venv by running requirements.txt

# Imports
import sys
import os
from pathlib import Path
from seg_mod_unet.helpers import window_predict, save_predictions, masks_to_submission, test_threshold
from seg_mod_unet.data_handling import extract_data, extract_data_test, extract_labels
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import segmentation_models as sm
sm.set_framework('tf.keras')

# Data paths
data_path = os.path.join(str(Path.cwd()), 'data')
train_path = os.path.join(data_path, 'training_final')
train_data_path = os.path.join(train_path, 'images/')
train_labels_path = os.path.join(train_path, 'groundtruth/')
test_images_path = os.path.join(data_path, 'testing/')

# Model paths
model_folder = os.path.join(str(Path.cwd()), 'models')
m1_path = os.path.join(model_folder, 'm1.h5')
m2_path = os.path.join(model_folder, 'm2.h5')
m3_path = os.path.join(model_folder, 'm3.h5')
m4_path = os.path.join(model_folder, 'm4.h5')
m5_path = os.path.join(model_folder, 'm5.h5')

# Filepaths to the five models' predictions on the test set
X1 = 'predictions/m1_pred.csv'
X2 = 'predictions/m2_pred.csv'
X3 = 'predictions/m3_pred.csv'
X4 = 'predictions/m4_pred.csv'
X5 = 'predictions/m5_pred.csv'

# Defining if model should be trained or not
TRAIN = sys.argv[1]

# Defining backbone for the model
BACKBONE = 'resnet34'

# Downloading preprocessing function for the model
preprocess_input = sm.get_preprocessing(BACKBONE)

# Defining loss, regularizer and optimizer for the model
optimizer = 'Adam'
kernel_regularizer = keras.regularizers.l2(1)
loss = sm.losses.bce_jaccard_loss

# Defining number of epochs and batch size
num_epcohs = 50
batch_size = 32

# Custom objects for the model
custom_objects = {'binary_crossentropy_plus_jaccard_loss': loss,
                  'iou_score': sm.metrics.iou_score, 'f1-score': sm.metrics.FScore()}

# Defining threhshold for attributing patch as road for each model
thresholds = [0.13, 0.06, 0.08, 0.01, 0.04]


def main():
    if TRAIN == 'True':
        # Extracting the data and masks
        print(sys.argv[1])
        x = extract_data(train_data_path)
        y = extract_labels(train_labels_path)

        # Training 5 seperate models
        for i in range(0, 5):
            print("%d. iteration" % (i+1))

            # Splitting the dataset into two, one training set and one validation set
            x_val, y_val = x[340*i:340*(i+1)], y[340*i:340*(i+1)]
            x_train, y_train = x[np.isin(np.arange(len(x)), np.arange(
                340*0, 340*(0+1)), invert=True)], y[np.isin(np.arange(len(y)), np.arange(340*0, 340*(0+1)), invert=True)]

            # Preprocessing training and validation data
            x_train = preprocess_input(x_train)
            x_val = preprocess_input(x_val)

            # Defining model, using 'imagenet' as weights to converge faster
            model = sm.Unet(BACKBONE, encoder_weights='imagenet',
                            input_shape=(256, 256, 3))

            # Adding  L2 kernel regularizer
            sm.utils.set_regularization(
                model, kernel_regularizer=keras.regularizers.l2(1))

            # Compiling the model using Adam optimizer and Binary Cross Entropy with Jaccard loss
            model.compile(
                'Adam',
                loss=sm.losses.bce_jaccard_loss,
                metrics=[sm.metrics.iou_score,
                         sm.metrics.FScore(), 'accuracy'],
            )

            # Saving the model thats scores best on the validation data
            callbacks = [keras.callbacks.ModelCheckpoint(
                "models/m%d.h5" % (i+1), save_best_only=True)]
            print("Training model %d\n" % (i+1))

            # Training the model for 50 epochs with batch size = 32
            history = model.fit(x=x_train, y=y_train,
                                epochs=1, batch_size=32,
                                callbacks=callbacks,
                                validation_data=(x_val, y_val)
                                )

            # Testing the model and finding optimal threshold
            model = load_model('models/m%d.h5' %
                               (i+1), custom_objects=custom_objects)

            # Generating predictions on validation set
            y_pred = model.predict(x_val)

            # Finding optimal threshold on validation set
            thr = test_threshold(y_pred, y_val, 0, 1)

            # Adding the optimal threshold to the thresholds array
            thresholds[i] = thr

    # Loading models
    print('Loading models')
    m1 = load_model(m1_path, custom_objects=custom_objects)
    m2 = load_model(m2_path, custom_objects=custom_objects)
    m3 = load_model(m3_path, custom_objects=custom_objects)
    m4 = load_model(m4_path, custom_objects=custom_objects)
    m5 = load_model(m5_path, custom_objects=custom_objects)

    models = [m1, m2, m3, m4, m5]
    print('Loading test images')
    
    # Loading test images
    test_images = extract_data_test(test_images_path)

    # Preprocessing test images
    test_images = preprocess_input(test_images)

    for i in range(len(models)):
        # Generating predictions for the test images
        print('Predicting for model %i' % (i+1))
        results = []
        for img in test_images:
            results.append(window_predict(img, models[i]))

        # Generating and saving the prediction masks for the testset
        for k in range(1, len(results)+1):
            save_predictions(results[k-1], 'test%d' % k)

        # Generating the prediction file for the test set
        submission_filename = 'predictions/m%d_pred.csv' % (i+1)
        image_filenames = []
        for j in range(1, 51):
            image_filename = 'predictions/test%d.png' % j
            image_filenames.append(image_filename)
        masks_to_submission(submission_filename,
                            thresholds[i], *image_filenames)
        print("Finished predicting model %d\n" % (i+1))

    # Reading the models' prediction into five dataframes
    print("Creating ensemble predictions")

    df1 = pd.read_csv(X1)
    df1 = df1.set_index(['id'])
    df1 = df1.rename({'prediction': 'p1'}, axis=1)

    df2 = pd.read_csv(X2)
    df2 = df2.set_index(['id'])
    df2 = df2.rename({'prediction': 'p2'}, axis=1)

    df3 = pd.read_csv(X3)
    df3 = df3.set_index(['id'])
    df3 = df3.rename({'prediction': 'p3'}, axis=1)

    df4 = pd.read_csv(X4)
    df4 = df4.set_index(['id'])
    df4 = df4.rename({'prediction': 'p4'}, axis=1)

    df5 = pd.read_csv(X5)
    df5 = df5.set_index(['id'])
    df5 = df5.rename({'prediction': 'p5'}, axis=1)

    # Dataframe containing the prediction of all models for each patch
    df = pd.concat([df1, df2, df3, df4, df5], axis=1)

    # Generating predictions, predicting road if all models predict road
    df['prediction'] = df.apply(lambda x: 1 if np.sum(x) > 4 else 0, axis=1)

    # Extracting only the prediction column
    df = df['prediction']

    # Writing predictions to csv
    df.to_csv('predictions/ensemble.csv')
    print("Prediction succeeded")


if __name__ == "__main__":
    main()
