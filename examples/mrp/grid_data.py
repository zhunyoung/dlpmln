import numpy as np 
import random
from numpy.random import permutation
import sys

edges_number = 24
grid_size = 4

def to_one_hot(dense, n, inv=False):
	one_hot = np.zeros(n)
	one_hot[dense] = 1
	if inv:
		one_hot = (one_hot + 1) % 2
	return one_hot
class GridProbData():
	def __init__(self, file):
		with open(file) as f:
			d = f.readlines()
			input_data = []
			output_data = []
			for data in d:
				data = data.strip().split(" ")
				two_point = [int(float(x)) for x in data[:2]]
				two_point = to_one_hot(two_point, grid_size*grid_size)
				prob = [float(x) for x in data[2:2+edges_number]]
				labels = [float(x) for x in data[2+edges_number:]]
				
				input_data.append(np.concatenate((np.array(prob),two_point)))
				output_data.append(labels)
		length = len(input_data)
		input_data = np.array(input_data)
		output_data = np.array(output_data)
		print("there are {} data in total, 60% training data, 20% validation data, 20% testing data!")
		self.train_data = input_data[:int(length*0.6)]
		self.train_labels = output_data[:int(length*0.6)]
		self.valid_data = input_data[int(length*0.6):int(length*0.8)]
		self.valid_labels = output_data[int(length*0.6):int(length*0.8)]
		self.test_data = input_data[int(length*0.8):]
		self.test_labels = output_data[int(length*0.8):]