# CS-433 - PROJECT 2 - ML_Vikings

## Description
This project is the second of two in the EPFL course "CS-433 Machine Learning". The aim of the project is solve the AICrowd challenge Road Segmenation hosted by MLO EPFL. 

A complete challenge description as well as the training and test data provided, can be found at:\
https://www.aicrowd.com/challenges/epfl-ml-road-segmentation

The repository is structured as follows:


## Installing and running the code

Start by pulling from the github as follows:\
`$ git clone https://github.com/CS-433/ml-project-2-ml_vikings.git` \
`$ cd ml-project-2-ml_vikings`

The project uses the following external libraries
 * `segmentation_models`
 * TensorFlow
 * Keras
 * SciKit Learn

If these libraries are not currently installed in your environment, they can be installed installing the requirements.txt file as follows:
`$ pip install -r requirements.txt`

There are two main ways to run the code. We reccommend using the trained models, i.e. load the models from memory to test and predict, but the model can also be trained from scratch. Below, the instructions for each of the options follows:

**Trained models**

To run the project, you have to download the trained model files from https://drive.google.com/drive/folders/1o5Rg-aVe2lkL_pcW1sLoRvFd2xuay8hn?usp=sharing as the files with the model parameters are too large to be pushed to GitHub. After downloading the files, put the folder into the repository.

After navigating to the project directry, you might run the following command to run the model without training from scratch:
`python3 run.py False`

For training from scratch, you can run:
`python3 run.py True`

This will yield a prediction file `ensemble.csv`.

**Train model from scratch**

If you want to train the model from scratch, we reccommend using Google Colab as the model is computionally expensive to train. Other GPU services can also be utlized, but we provide the instructions for Colab.

1. Create a shortcut on the "desktop" of your Google Drive of this folder: https://drive.google.com/drive/folders/1-3Rjf92mn-WFVhhwdblE1-VnflLfw74X?usp=sharing
2. Find the file `final_model.ipynb` in the seg_model_unet folder in this repository and upload it to Google Colab
3. Run the code cells in the notebook in chronological order and a prediction file named `ensemble.ipynb` will appear in the file directory on the left

## Overview of folder structure

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
│   └── training_double
│       ├── groundtruth
│       │   ├── 10-split
│       │   └── 90-split
│       └── images
│           ├── 10-split
│           └── 90-split
├── baseline
│   ├── Logistic.ipynb
│   ├── baseline.py
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
│   ├── ensemble_model.ipynb
│   ├── final_model.ipynb
│   ├── helpers.py
│   └── mask_to_submission.py
│ 
├── predictions
│   └── .
│ 
├── Ensemble.ipynb
├── augmentation.py
├── run.py
├── story.ipynb
└── requirements.txt
```

## Overview of key files and folders

The folder `data/` contains the original train and test data as well as the 2 augmented setups (`training` and `training_double` respectively) with a 90/10 split.

The folder `baseline/` contains the implementation and the scripts to train the baseline CNN and LogReg. 

The folder `unet_classical/` contains the implementation and the scripts to train the simple U-net.

The folder `seg_mod_unet/` contains the implementation and the scripts to train the ensemble of U-nets with the resnet backbone. 

The script `augmentation.py` contains the functions used to generate augmentations of the original data set.

The script `run.py` generates the submission uploaded to AICrowd with submission ID [#167620](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/submissions/167620). This is the submission which recieved an F1-score of 0.901 and are referring to in the project report.

The file `report.pdf` contains our report documenting the project. Complimentary to this is the `story.ipynb` notebook. 



## Authors
Sigurd Kampevold Johanson - sigurd.johanson@epfl.ch\
Benjamin Rike - benjamin.rike@epfl.ch\
Nikolai Beck Jensen - nikolai.jensen@epfl.ch
