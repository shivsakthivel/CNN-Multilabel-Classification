# DSC-180B-Project

# DSC 180A - Exploring the viability of Convolutional Neural Networks (CNNs) on a multi-label classification task to detect radiographic outliers

## Task
An implementation of a Convolutional Neural Network (CNN) multi-label classifier that takes in chest radiograph images as and outputs their corresponding predicted labels for detecting pulmonary edema and pleural effusion.

## Retrieving the Data for this project
The data available for this project came in the form of DICOM files stored on a Google Cloud instance, with the entire database being of size 4 TB. Therefore, this GitHub repository contains exploratory notebooks, walking through the data access and model training process. However, the scripts associated with the repository only replicate the overall results of the project.

## Building the project results using run.py

Run `python run.py test`

This will produce an output file with the visualizations associated with the model loss and accuracy on each of the predicted labels.

## Other Files in the Repository

A large section of the code and model development was done in Jupyter Notebooks, so that is a good resource for how the model was developed from the given data files.
