import os
import matplotlib.image as mpimg
import random
import numpy as np
import math
from tensorflow import keras

def create_data_model(split=0.8, batch_size=8):
    # Load file names
    data_dir = 'data/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/' 
    input_img_paths, target_img_paths = extract_data_unet(train_data_filename, train_labels_filename)

    # Split our img paths into a training and a validation set
    val_samples = math.ceil((1-split)*len(input_img_paths))
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = SatImageSequence(x_set=train_input_img_paths, y_set=train_target_img_paths,batch_size=batch_size)
    val_gen = SatImageSequence(x_set=val_input_img_paths, y_set=val_target_img_paths,batch_size=batch_size)

    return train_gen, val_gen
    


class SatImageSequence(keras.utils.Sequence):
    """
    From tensorflow documentation: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """
    def __init__(self, x_set, y_set, batch_size=16):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        # x values
        x_gen = np.array([mpimg.imread(file_name) for file_name in batch_x])

        # y values. descale augmented pictures from 3 dimensions to 1
        y_gen = np.array([mpimg.imread(file_name)[:,:,0] if len(mpimg.imread(file_name).shape)==3 else mpimg.imread(file_name) for file_name in batch_y])
        return x_gen, y_gen


def extract_data_unet(filename_img, filename_label):
    """ (ETH modified by ML_vikings) Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].

    Parameters
    ----------
    filename: string
        The name of the image file

    Returns
    -------
    data: ndarray
        A numpy array containting the images
    """
    imgs = []
    labels = []
    for i in range(1, 100 + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename_img + imageid + ".png"
        label_filename = filename_label + imageid + ".png"
        if os.path.isfile(image_filename) and os.path.isfile(label_filename):
            #print('Loading ' + image_filename)
            #img = mpimg.imread(image_filename)
            imgs.append(image_filename)
            #lab = mpimg.imread(label_filename)
            labels.append(label_filename)
        else:
            print('File ' + image_filename + ' does not exist')
        
        for j in range(16):
            imageid = "satImage_%.3d" % i
            imageid += '_Aug%.2d' % j
            image_filename = filename_img + 'images_aug/' + imageid + '.png'
            label_filename = filename_label + 'groundtruth_aug/' + imageid + '.png'
            if os.path.isfile(image_filename) and os.path.isfile(label_filename):
                #print('Loading ' + image_filename)
                #img = mpimg.imread(image_filename)
                imgs.append(image_filename)
                #lab = mpimg.imread(label_filename)
                labels.append(label_filename)
            else:
                print('File ' + image_filename + ' does not exist')

    return imgs,labels