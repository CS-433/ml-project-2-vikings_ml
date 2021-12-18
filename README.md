# CS-433 - PROJECT 2 - ML_Vikings

## Description
This project is the second of two in the EPFL course "CS-433 Machine Learning". The aim of the project is solve the AICrowd challenge Road Segmenation hosted by MLO EPFL. Our best submission achieved a F1-score of 0.901 and can be found here: https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/submissions/167620. 

A complete challenge description as well as the training and test data provided, can be found at:\
https://www.aicrowd.com/challenges/epfl-ml-road-segmentation

The repository is structured as follows:


## Installing and running the code

Start by pulling from the github as follows:\
`git clone https://github.com/CS-433/ml-project-2-ml_vikings.git` \
`cd ml-project-1-ml_vikings`

or download the delivered `.zip`-file.

To run the project, you have to download the trained model files from **google colab link**, as the files with the model parameters are too large to be pushed to GitHub. 

After navigating to the project directry, you might run the following commands:
`cd src`\
`python3 run.py`

This will yield a prediction file `______.csv`.

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

## Overview of files

The folder `Linear/`contains helper functions for our linear regression models. These are mainly for used for our ridge regression models.

The folder `Logistic/` contains helper functions for our logistic regression model.

`feature_processing.py` contains helper functions for feature processing, where some are general while other are specific for our logistic regression model and our ridge regression models.

`helpers.py` contains general helper functions for the project.

`loss.py` contains a function for computing the loss of model with a desired loss function.

`predict.py` contains functions for making predictions.

`proj1_helpers.py` contains helper functions for the project, mainly those that are provided by the course staff.

`opti_hyperparameters.ipynb` contains functions that make visualiziations of our choices of hyperparameters.

The notebooks `logistic_regression.ipynb`, `ridge_regression_three.ipynb` and `ridge_regression.ipynb` contains feature processing and training of the models, as well as prediction generation.

The script `run.py` generates the submission uploaded to AICrowd with submission ID [#164073](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/submissions/164073). This is the submission we are referring to in the project report.

The file `report.pdf` contains our report documenting the project.

## Authors

Sigurd Kampevold Johanson - sigurd.johanson@epfl.ch\
Benjamin Rike - benjamin.rike@epfl.ch\
Nikolai Beck Jensen - nikolai.jensen@epfl.ch
