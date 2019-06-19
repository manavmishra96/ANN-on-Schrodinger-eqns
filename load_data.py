"""
Python program to extract data from the csv file and load them into the python program.
The data set is divided as follows:
train_x, train_y: m = 480
valid_x, valid_y: m = 120

no. of features: n = 127

These are the limited data-sets which I was able to generate after running the genpotential.py python snippet
(whose working still remains a mystery!)

Calling this function from this snippet would extract all the required data
and store them in python variables.
"""

import csv
import numpy as np 
import tensorflow as tf 

def load_data_nn():
	train_x = []; train_y = []
	valid_x = []; valid_y = []

	with open('Training_pot.csv','r') as csvfile:
		flurg = csv.reader(csvfile)
		for row in flurg:
			train_x.append([float(num) for num in row])

	with open('Training_wf.csv', 'r') as csvfile:
		flurg = csv.reader(csvfile)
		for row in flurg:
			train_y.append([float(num) for num in row])

	with open('Validation_pot.csv', 'r') as csvfile:
		flurg = csv.reader(csvfile)
		for row in flurg:
			valid_x.append([float(num) for num in row])

	with open('Validation_wf.csv', 'r') as csvfile:
		flurg = csv.reader(csvfile)
		for row in flurg:
			valid_y.append([float(num) for num in row])

	training_data = []
	for i in range(len(train_x)):
		t = (train_x[i], train_y[i])
		training_data.append(t)

	validation_data = []
	for j in range(len(valid_x)):
		v = (valid_x[j], valid_y[j])
		validation_data.append(v)

	return training_data, validation_data			

