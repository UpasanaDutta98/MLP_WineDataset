# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
 
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network


# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation


# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row): #row is one row of inputs from the training set. So this becomes the input to the first layer (input_layer)
	inputs = row
	# Then we have nested loop to iterate over each neuron of each layer of the network.
	for layer in network:
		new_inputs = [] # input list of each layer will be different
		for neuron in layer: ## the following block is run for each neuron of the network
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs ## so this will return the 'output' of the last layer, which is the final output of the forward propagation.



# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[0]-1] = 1
			#for i in range(len(expected)) :
			#	print(expected[i], outputs[i])
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, error=%.3f' % (epoch, sum_error))



# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)


# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:		# i.e for except the first layer, input will be the output of the neurons of the previous layer.
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
 


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected): # here network is the entire network and expected is a list like [0 1 0] or [0 0 1]
	for i in reversed(range(len(network))):  # so network[i] will be starting from the last layer of the network
		layer = network[i] 		 # type(layer) = list	
		errors = list() 		 # error is initialised as a separate list for each layer
		if i != len(network)-1:          # i.e when the iteration is not at the output layer
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]: # we sum over every neuron of the next layer
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)): # here len(layer) is 3 because our output layer will have 3 neurons for the wine dataset
				neuron = layer[j] # so neuron is a dictionary here
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j] 	  # so neuron is a dictionary here
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])



# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1	# because 1st index will be the class value, remaining index are the input
	n_outputs = len(set([row[0] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	#print(predictions)
	return predictions



# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))	#max(outputs) will be 1. We need the index of '1', which will be either 0, 1 or 2.
 

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]	#stats = A list of 14 lists, each of size 2.
	return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(1, len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()	# empty list
	dataset_copy = list(dataset)	# creating a copy of the dataset list
	fold_size = int(len(dataset) / n_folds)		# size of each fold will be (total size / number of folds)
	for i in range(n_folds):
		fold = list()	# fold will be a list of size fold_size
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy)) # generate a random index from 0 to len(dataset_copy) - 1
			fold.append(dataset_copy.pop(index)) # once a row is chosen at random, it is popped/removed out from the dataset_copy.
		dataset_split.append(fold)
		# len(dataset_copy)) decreases by 35 after each iteration.
	'''
	print(dataset_split)
	for i in range(len(dataset_split)) :
		print(len(dataset_split[i]))
		print(len(dataset_split[i][0]))
	'''
	return dataset_split # so dataset_split is a list of size 5, where at each index we have a list of 35 lists (35 random rows from the dataset and each row is present in the form of a list).

 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	#print(actual, predicted)
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 



# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args): #With *args, any number of extra arguments can be tacked on to your current formal parameters.
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:	# i.e for each list from the set of 5 lists
		train_set = list(folds) 
		train_set.remove(fold)	# all other lists except the testing list becomes the training set
		train_set = sum(train_set, []) #Removes the outermost listing of the train_set. So contents of the remaining 4 lists are merged.
		'''
		for example - [[2,3,4],[5,6,7],[1,8,9]] gets converted to [2, 3, 4, 5, 6, 7, 1, 8, 9]
		'''
		test_set = list()
		for row in fold:	# fold is the set of testing rows
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		''' Or we can just write test_set = fold '''
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[0]-1 for row in fold]	# row[0] will be either 1, 2 or 3. actual will be a list of class lables.
		print("\n\n\n\n\n\n")
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores 

  

 
# Test Backprop on Seeds dataset
seed()
# load and prepare data
filename = 'wine.csv'
dataset = load_csv(filename)	# this function converts the csv file to a list of lists and returns that list
for i in range(1,len(dataset[0])):
	str_column_to_float(dataset, i)
# convert class column to integers
#str_column_to_int(dataset, 0)
# normalize input variables

for i in range(len(dataset)):
	dataset[i][0] = int(dataset[i][0])

#print(dataset)	
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 15
n_hidden = 13
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.5f%%' % (sum(scores)/float(len(scores))))
