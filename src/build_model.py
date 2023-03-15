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

def get_model(target):
	resnet = tf.keras.applications.ResNet152V2(
	    include_top=False,
	    weights='imagenet',
	    input_shape=(512, 512, 3),
	    pooling = "avg",
	    classifier_activation = 'softmax'
	)

	# Freezing the pre-trained weights
	for layer in resnet.layers:
	    layer.trainable = False


	# Augmenting the final linear layer for each model specifically
	if 'edema' == target or 'effusion' == target:
		x = resnet.output
		x = tf.keras.layers.Dense(1024, activation='relu')(x)
		outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

	elif 'multilabel' == target:
		x = resnet.output
		x = tf.keras.layers.Dense(1024, activation='relu')(x)
		outputs = tf.keras.layers.Dense(2, activation='sigmoid')(x)

	elif 'multiclass' == target:
		x = resnet.output
		x = tf.keras.layers.Dense(1024, activation='relu')(x)
		outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

	model = Model(inputs=resnet.input, outputs=outputs)

	return model