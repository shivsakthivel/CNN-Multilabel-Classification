# Packages to import
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

# The code should be run within the DSMLP account, outside the private directory. 


def get_data(target):
	# Reading in the files 
	files = pd.read_csv('private/all_filepaths.csv')
	files_proper = files.apply(lambda x: x['path'][14:], axis = 1)
	files['path'] = files_proper

	if 'edema' == target:
		overall_data = files[['path', 'Edema']]
		overall_data['Edema'] = overall_data['Edema'].astype(str)
	if 'effusion' == target:
		overall_data = files[['path', 'Pleural Effusion']]
		overall_data['Pleural Effusion'] = overall_data['Pleural Effusion'].astype(str)
	if 'multilabel' == target:
		overall_data = files[['path', 'Edema', 'Pleural Effusion']]
		overall_data['Edema'] = overall_data['Edema'].astype(str)
		overall_data['Pleural Effusion'] = overall_data['Pleural Effusion'].astype(str)
	if 'multiclass' == target:
		files['labels'] = files.apply(lambda x: combined_labels(x), axis = 1)
		files['labels'] = files['labels'].astype(str)
		overall_data = files[['path', 'labels']]

	return overall_data

# Binarizer for the Multiclass Classifier
def combined_labels(row):
    to_return = []
    if row['Edema'] == 1.0:
        to_return.append(1)
    else:
        to_return.append(0)
        
    if row['Pleural Effusion'] == 1.0:
        to_return.append(1)
    else:
        to_return.append(0)
        
    if to_return == [0, 0]:
        return 0
    elif to_return == [0, 1]:
        return 1
    elif to_return == [1, 0]:
        return 2
    elif to_return == [1, 1]:
        return 3

def image_generators(target, train_data, val_data, test_data):

	# Define the ImageDataGenerator
	train_gen = ImageDataGenerator(rescale=1./255)
	val_gen = ImageDataGenerator(rescale=1./255)
	test_gen = ImageDataGenerator(rescale=1./255)

	# Use flow_from_dataframe method to create the generator (depending on the target)
	if 'edema' == target:
		train_images = train_gen.flow_from_dataframe(
		    dataframe=train_data,
		    directory=None,
		    x_col="path",
		    y_col="Edema",
		    target_size=(512, 512),
		    batch_size=32,
		    class_mode="binary",
		    shuffle=True
		)

		val_images = val_gen.flow_from_dataframe(
		    dataframe=val_data,
		    directory=None,
		    x_col="path",
		    y_col="Edema",
		    target_size=(512, 512),
		    batch_size=32,
		    class_mode="binary",
		    shuffle=True
		)
		test_images = test_gen.flow_from_dataframe(
		    dataframe=test_data,
		    directory=None,
		    x_col="path",
		    y_col="Edema",
		    target_size=(512, 512),
		    batch_size=32,
		    class_mode="binary",
		    shuffle=False
		)

	if 'effusion' == target:
		train_images = train_gen.flow_from_dataframe(
		    dataframe=train_data,
		    directory=None,
		    x_col="path",
		    y_col="Pleural Effusion",
		    target_size=(512, 512),
		    batch_size=32,
		    class_mode="binary",
		    shuffle=True
		)

		val_images = val_gen.flow_from_dataframe(
		    dataframe=val_data,
		    directory=None,
		    x_col="path",
		    y_col="Pleural Effusion",
		    target_size=(512, 512),
		    batch_size=32,
		    class_mode="binary",
		    shuffle=True
		)
		test_images = test_gen.flow_from_dataframe(
		    dataframe=test_data,
		    directory=None,
		    x_col="path",
		    y_col="Pleural Effusion",
		    target_size=(512, 512),
		    batch_size=32,
		    class_mode="binary",
		    shuffle=False
		)

	if 'multilabel' == target:
		train_images = train_gen.flow_from_dataframe(
		    dataframe=train_data,
		    directory=None,
		    x_col="path",
		    y_col=['Edema', 'Pleural Effusion'],
		    target_size=(512, 512),
		    batch_size=32,
		    class_mode="raw",
		    shuffle=True
		)

		val_images = val_gen.flow_from_dataframe(
		    dataframe=val_data,
		    directory=None,
		    x_col="path",
		    y_col=['Edema', 'Pleural Effusion'],
		    target_size=(512, 512),
		    batch_size=32,
		    class_mode="raw",
		    shuffle=True
		)
		test_images = test_gen.flow_from_dataframe(
		    dataframe=test_data,
		    directory=None,
		    x_col="path",
		    y_col=['Edema', 'Pleural Effusion'],
		    target_size=(512, 512),
		    batch_size=32,
		    class_mode="raw",
		    shuffle=False
		)

	if 'multiclass' == target:
		train_images = train_gen.flow_from_dataframe(
		    dataframe=train_data,
		    directory=None,
		    x_col="path",
		    y_col="labels",
		    target_size=(512, 512),
		    batch_size=32,
		    class_mode="categorical",
		    shuffle=True
		)

		val_images = val_gen.flow_from_dataframe(
		    dataframe=val_data,
		    directory=None,
		    x_col="path",
		    y_col="labels",
		    target_size=(512, 512),
		    batch_size=32,
		    class_mode="categorical",
		    shuffle=True
		)
		test_images = test_gen.flow_from_dataframe(
		    dataframe=test_data,
		    directory=None,
		    x_col="path",
		    y_col="labels",
		    target_size=(512, 512),
		    batch_size=32,
		    class_mode="categorical",
		    shuffle=False
		)

	return train_images, val_images, test_images
