# Packages to import
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

def model_evaluation(history, model, test_images, target):
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	plt.plot(np.arange(len(loss)), loss, label='Training')
	plt.plot(np.arange(len(val_loss)), val_loss, label='Validation')
	plt.xlabel('Epochs')
	plt.ylabel('Cross Entropy Loss')
	plt.title("Training and Validation Plots")
	plt.legend()

	plt.plot(np.arange(len(acc)), acc, label='Training')
	plt.plot(np.arange(len(val_acc)), val_acc, label='Validation')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title("Training and Validation Plots")
	plt.legend()

	predictions = model.predict(test_images)
	true_classes = np.array(test_images.labels)
	if 'edema' == target or 'effusion' == target:
		predicted_classes = np.where(predictions < 0.5, 0, 1)
		cf_matrix = confusion_matrix(true_classes, predicted_classes)
		ax = sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues')
		ax.set_title('Confusion Matrix \n\n');
		ax.set_xlabel('\nPredicted Values')
		ax.set_ylabel('Actual Values ');

		ax.xaxis.set_ticklabels(['False','True'])
		ax.yaxis.set_ticklabels(['False','True'])

		plt.show()

		fpr, tpr, threshold = roc_curve(true_classes, predictions, drop_intermediate = False)
		roc_auc = roc_auc_score(true_classes, predictions)

		plt.figure(1)
		plt.plot(fpr, tpr, label ='ROC(area = {:.3f})'.format(roc_auc))
		plt.xlabel('False positive rate')
		plt.ylabel('True positive rate')
		plt.title(f'{target} ROC curve')
		plt.legend(loc = 'best')
		plt.show()
		

