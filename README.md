# DSC 180B - Exploring the viability of Convolutional Neural Networks (CNNs) on a multi-label classification task to detect radiographic outliers

## Task
An implementation of a Convolutional Neural Network (CNN) multi-label classifier that takes in chest radiograph images as and outputs their corresponding predicted labels for detecting pulmonary edema and pleural effusion.

## Retrieving the Data for this project
The data available for this project came in the form of DICOM files stored on a Google Cloud instance (credentialed access only), with the entire database being of size 4 TB. The required credentialing can be obtained [here](https://physionet.org/content/mimic-cxr/2.0.0/). For the purposes of this project and accessing the DSMLP resources, the source radiograph images had to be manually downloaded in batches and transferred onto the teams drive on DSMLP. 

The scripts associated with this repository, therefore, assume that the user has the required access to the data files, with the required filepaths relative to the directory in which they were developed. However, this GitHub repository contains exploratory notebooks, walking through the data access and model training process, and covers examples of the obtained results. Specifically, the notebook `Single-Var-Model-Edema.ipynb` is a comprehensive exploration of one of the single label binary classifiers developed for this project. The model build and evaluation techniques for the other models developed in this project largely follow a similar process.

## Build and Run
- To run the single label Pulmonary Edema classifier run `python main.py edema`.
- To run the single label Pleural Effusion classifier run `python main.py effusion`.
- To run the multi-label classifier run `python main.py multilabel`.
- To run the multi-class classifier run `python main.py multiclass`.

## Requirements
The dependencies required for this project can be installed by running `pip install -r requirements.txt`.

## Other Notes
If running the code on this repository, the DSMLP instance should be launched with GPU to ensure that the files run efficiently (The project was developed using tensorflow).
