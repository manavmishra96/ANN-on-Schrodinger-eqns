from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import pandas as pd

import numpy as np
from load_data import load_data_nn
import random as ra
import matplotlib.pyplot as plt


training_data, test_data = load_data_nn()

# Shuffle the training set
ra.shuffle(training_data)
ra.shuffle(test_data)

#Unpacking the tuples
train_x = []; train_y = []
for ele in training_data:
	c,d = ele
	train_x.append(c)
	train_y.append(d)
train_x = np.array(train_x); train_y = np.array(train_y)

test_x = []; test_y = []
for ele in test_data:
	c,d = ele
	test_x.append(c)
	test_y.append(d)
test_x = np.array(test_x); test_y = np.array(test_y)

print("Training set: {}".format(np.array(train_x).shape))  
print("Testing set:  {}".format(np.array(test_y).shape))

column_names = [i+1 for i in range(127)]

df = pd.DataFrame(train_x, columns=column_names)
#print (df.head())

#Normalizing the features
mean = train_x.mean(axis=0)
std = train_x.std(axis=0)
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std


#Function to build my neural network model
def build_model():
	model = keras.models.Sequential([keras.layers.Dense(200, kernel_regularizer=keras.regularizers.l2(0.001),\
		activation=tf.nn.relu,input_dim=127),\
		keras.layers.Dense(200,kernel_regularizer=keras.regularizers.l2(0.001),activation=tf.nn.relu),\
		keras.layers.Dense(127)])
	optimizer = tf.train.RMSPropOptimizer(0.001)
	model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

	return model

model = build_model()
#print(model.summary())

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

#The model is trained for n epochs, and record the training and validation accuracy in the history object.
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1500
# Store training stats
history = model.fit(train_x, train_y, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])


#Visualize the model's training progress using the stats stored in the history object.
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Test loss')
  plt.legend()
  plt.ylim([0, 5])
  plt.show()

plot_history(history)

#We want to see how the model performs on the test data
[loss, mae] = model.evaluate(train_x, train_y, verbose=0)
print("\n\nTraining set Mean Abs Error: {:7.2f}".format(mae))

[loss, mae] = model.evaluate(test_x, test_y, verbose=0)
print("Testing set Mean Abs Error: {:7.2f}".format(mae))

def plot_graph(test_predictions, test_y, test_x):
	plt.ylabel('Wavefunction')
	plt.xlabel('Features')
	plt.plot(column_names,np.transpose(test_predictions), label='Predicted value')
	plt.plot(column_names, test_y, label='Actual value')
	plt.plot(column_names, test_x, label='Potential')
	plt.legend()
	plt.show()

for i in range(5):
	a = np.array(test_x[i])[np.newaxis]
	test_predictions = model.predict(a)
	plot_graph(test_predictions, test_y[i], test_x[i])
