# DSC 180B - Exploring the viability of Convolutional Neural Networks (CNNs) on a multi-label classification task to detect radiographic outliers

## Task
An implementation of a Convolutional Neural Network (CNN) multi-label classifier that takes in chest radiograph images as and outputs their corresponding predicted labels for detecting pulmonary edema and pleural effusion.

## Retrieving the Data for this project
The data available for this project came in the form of DICOM files stored on a Google Cloud instance (credentialed access only), with the entire database being of size 4 TB. The required credentialing can be obtained [here](https://physionet.org/content/mimic-cxr/2.0.0/). The scripts associated with this repository, therefore, assume that the user has the required access to the data files. However, this GitHub repository contains exploratory notebooks, walking through the data access and model training process, and covers examples of the obtained results.

## Requirements
The dependencies required for this project can be installed by running `pip install -r requirements.txt`.
