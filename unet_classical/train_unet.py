""" Training of our original UNet. """

from Unet import UNet
from data_handling_unet import create_data_model
from tensorflow import keras


if __name__ == "__main__":

    # Loading Unet NN architecture
    model = UNet().get_model(image_shape=(400,400,3))

    # Creating data model. Split= % of data to be used as training. T0DØ: batchsize param + augmented pictures
    train_gen, val_gen = create_data_model(split=0.2)

    # Training and saving best model
    epochs = 3
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    callbacks = [keras.callbacks.ModelCheckpoint("Unet_roadsegment1.h5", save_best_only=True)]
    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

