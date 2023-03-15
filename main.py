# To Import
import sys
import warnings
import json
import os
import tensorflow as tf

from src.make_dataset import *
from src.build_model import *
from src.evaluation import *


def main(targets):

	# Ensure GPU Usage
	print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

	# Dataframe of Filepaths
	target = targets[0]
	overall_data = get_data(target)

	# Train, Validation, Test Split
	train_data, test_data = train_test_split(overall_data, test_size=0.2, random_state=42)
	val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

	train_images, val_images, test_images = image_generators(target)

	model = get_model(target)

	if 'multiclass' == target:
		# Compute class weights
		class_counts_max = 0
		class_weights = {}
		for i in range(4):
		    if np.sum(train_data['labels'] == str(i)) > class_counts_max:
		        class_counts_max = np.sum(train_data['labels'] == str(i))
		for i in range(4):
		    class_weights[i] = class_counts_max / np.sum(train_data['labels'] == str(i))


	# First setting model training for all the models (All ResNet layers frozen)

	if 'multiclass' == target:
		opt = tf.keras.optimizers.Adam(learning_rate = 0.00001)

		model.compile(
		    optimizer=opt,
		    loss='categorical_crossentropy',
		    metrics=['accuracy']
		)

		history = model.fit(
		    train_images,
		    validation_data=val_images,
		    epochs=10,
		    class_weight=class_weights,
		    callbacks=[
		        tf.keras.callbacks.EarlyStopping(
		            monitor='val_loss',
		            patience=3,
		            restore_best_weights=True
		        )
		    ]
		)

	elif 'edema' == target or 'effusion' == target or 'multilabel' == target:
		opt = tf.keras.optimizers.Adam(learning_rate = 0.00001)

		model.compile(
		    optimizer=opt,
		    loss='binary_crossentropy',
		    metrics=['accuracy']
		)

		history = model.fit(
		    train_images,
		    validation_data=val_images,
		    epochs=10,
		    callbacks=[
		        tf.keras.callbacks.EarlyStopping(
		            monitor='val_loss',
		            patience=3,
		            restore_best_weights=True
		        )
		    ]
		)

	model_evaluation(history, model, test_images, target)


if __name__ == '__main__':
    targets = sys.argv[1:]
    warnings.filterwarnings('ignore')
    main(targets)