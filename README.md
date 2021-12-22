# CS-433 - PROJECT 2 - ML_Vikings

## Description

This project is the second of two in the EPFL course "CS-433 Machine Learning". The project aims to solve the AICrowd challenge Road Segmentation hosted by MLO EPFL. The challenge aims to train machine learning models that segment roads from the background in satellite images.

A complete challenge description, as well as the training and test data provided, can be found [here](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation).

We recommend reading the report and the corresponding `story.ipynb` notebook to best understand the implementations we did during the competition. The report outlines our implementations' theoretical background and results, while the `story.ipynb` shows our progress and predictions.

## Overview of Folder Structure

```
.
├── data
│   ├── original
│   │   ├── groundtruth
│   │   └── images
│   ├── testing
│   │   └── .
│   ├── training
│   │   ├── groundtruth
│   │   │   ├── 10-split
│   │   │   └── 90-split
│   │   └── images
│   │       ├── 10-split
│   │       └── 90-split
│   ├── training_double
│   │   ├── groundtruth
│   │   │   ├── 10-split
│   │   │   └── 90-split
│   │   └── images
│   │       ├── 10-split
│   │       └── 90-split
│   └── training_final
│       ├── groundtruth
│       │   ├── 10-split
│       │   └── 90-split
│       └── images
│           ├── 10-split
│           └── 90-split
│
├── baseline
│   ├── logistic.ipynb
│   ├── baseline.py
│   ├── handling_images.py
│   └── helpers.py
│
├── unet_classical
│   ├── Unet.py
│   ├── data_handling_unet.py
│   ├── run_unet_nb.ipynb
│   └── train_unet.py
│
├── seg_mod_unet
│   ├── data_handling.py
│   ├── final_model.ipynb
│   └── helpers.py
│
├── predictions
│   └── .
│
├── augmentation.py
├── run.py
├── story.ipynb
└── requirements.txt
```

## Installing and Running the Code

Start by pulling from GitHub as follows:\
`$ git clone https://github.com/CS-433/ml-project-2-ml_vikings.git` \
`$ cd ml-project-2-ml_vikings`

The project uses the following external libraries

- `segmentation_models`
- TensorFlow
- Keras
- SciKit Learn

If these libraries are not currently installed in your environment, they can be installed installing the `requirements.txt` file with the following command:\
`$ pip install -r requirements.txt`

There are two main ways to run the code. We recommend using the pre-trained models, i.e. load the models from memory to test and predict, but the model can also be trained from scratch. Below, the instructions for each of the options follows:

**Trained models**

To run the project, you have to download the trained model files from https://drive.google.com/drive/folders/1o5Rg-aVe2lkL_pcW1sLoRvFd2xuay8hn?usp=sharing as the files with the model parameters are too large to be pushed to GitHub. After downloading the files, put the folder into the repository.

After navigating to the project directry, you might run the following command to run the model without training from scratch:\
`$ python3 run.py False`

This will yield a prediction file `ensemble.csv`.

**Train model from scratch**

If you want to train the model from scratch, we reccommend using Google Colab as the model is computionally expensive to train. Other GPU services can also be utlized, but we provide the instructions for Colab.

1. Create a shortcut on the "desktop" of your Google Drive of this folder: https://drive.google.com/drive/folders/1-3Rjf92mn-WFVhhwdblE1-VnflLfw74X?usp=sharing
2. Find the file `final_model.ipynb` in the seg_model_unet folder in this repository and upload it to Google Colab
3. Run the code cells in the notebook in chronological order and a prediction file named `ensemble.ipynb` will appear in the file directory on the left

## Overview of Key Files and Folders

The folder `data/` contains the original train and test data as well as the two augmented setups, `training` and `training_double`, with a 90/10 split, and finally `training_final` which is similar to the `training` folder, but all crops is of size 256x256 without resizing and there is no 90/10 split. All the augmented setups contains an `images` and a `groundtruth` folder.

The folder `baseline/` contains the implementation and the scripts to train the baseline CNN and the simple logistic regression:
* `baseline.py` contains the code for the baseline CNN provided by the course staff, and is credited to Aurelien Lucchi (ETH).
* `handling_images.py` provides functions for handling images in `baseline.py`, and should also be credited to Lucchi.
* `logistic.ipynb` contains our implementation of a logistic regression baseline.
* `helpers.py` provides several helper functions for the logistic regression that are provided by the course staff.

The folder `unet_classical/` contains the implementation and the scripts to train our first UNet implementation. This UNet is implemented by ourselves using TensorFlow.
* `data_handling_unet.ipynb` contains helper functions for data handling related to our UNet model.
* `Unet.py` contains our UNet model as well as the LadderNet expansion.
* `train_unet.py` contains training of our UNet.
* `run_unet_nb.ipynb` contains training of our UNet.

The folder `seg_mod_unet/` contains the implementation and the scripts to train the ensemble of UNets with the ResNet34 backbone. These UNets are implemented by using the `segmentation_models` library.
* `data_handling.py` contains helper functions for handling images provided by the course staff and ETH.
* `final_model.ipynb` handles the data processing, modelling, predictions, and post-processing for the Unet with ResNet34 architecture.
* `helpers.py` contains helper functions utilized in `final_model.ipynb`.

The script `augmentation.py` contains the functions used to generate augmentations of the original data set.

The script `run.py` generates the submission uploaded to AICrowd with submission ID [#167620](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/submissions/167620). This is the submission which recieved an F1-score of 0.901 and that is referred to in the project report.

The file `report.pdf` contains our report documenting the project. Complimentary to this is the `story.ipynb` notebook, where we try to show our experimenting throughout the project.

### Comments on `segmentation_models`

[`segmentation_models`](https://github.com/qubvel/segmentation_models) is a Python library consisting of implemented Neural Networks for image segmentation provided by Pavel Yakubovskiy. The models are based on Keras and TensorFlow. The library is easy to use with an understandable API and provides different NN architectures with several available backbones, e.g., architectures for the encoding layers. We have mainly utilized the UNet architecture with a ResNet34 backbone, but have also experimented with different architectures and backbones. The complete documentation for the library can be found [here](https://segmentation-models.readthedocs.io/en/latest/index.html).

## Authors

Sigurd Kampevold Johanson - sigurd.johanson@epfl.ch\
Benjamin Rike - benjamin.rike@epfl.ch\
Nikolai Beck Jensen - nikolai.jensen@epfl.ch
