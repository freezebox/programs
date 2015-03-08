#Simple Multilayer Perceptron Neural Network
#with Stochastic Gradient Descent
from random import uniform, randint
from math import exp, floor

#Definitions
#******************************************************************************
#Activation function for each neuron
def activation(weighted_input):
	return ((exp(weighted_input) - exp(-weighted_input)) / (exp(weighted_input) + exp(-weighted_input)))
	
#Activation function derivative
def act_derivative(weighted_input):
	return (4 / (exp(2 * weighted_input) + exp(-2 * weighted_input) + 2))

#Function to apply act_derivative to a matrix's elements individually
def act_derivative_to_matrix(weighted_input_matrix):
	for i in range(len(weighted_input_matrix)):
		for j in range(len(weighted_input_matrix[i])):
			weighted_input_matrix[i][j] = act_derivative(weighted_input_matrix[i][j])
	return weighted_input_matrix
	
#Function for dot product between two vectors
def dot_product(vector_1, vector_2):
	if not len(vector_1) == len(vector_2):
		print("Vectors wrong length")
		raise SystemExit
	return sum(vector_1[i] * vector_2[i] for i in range(len(vector_1)))
	
#Matrix transpose
def transpose(matrix):
	place_holder = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
	return place_holder
	
#Matrix multiplication
def matrix_multiply(matrix_1, matrix_2):
	place_holder_1 = transpose(matrix_2)
	place_holder_2 = [[] for i in range(len(matrix_1))]
	for i in range(len(matrix_1)):
		for j in range(len(place_holder_1)):
			place_holder_2[i].append(dot_product(matrix_1[i], place_holder_1[j]))
	return place_holder_2
	
#Hadamard Product
def hadamard_product(matrix_1, matrix_2):
	if not len(matrix_1) == len(matrix_2):
		print("Matrices wrong size")
		raise SystemExit
	for i in range(len(matrix_1)):
		if not len(matrix_1[i]) == len(matrix_2[i]):
			print("Matrices wrong size")
			raise SystemExit
	result = [[] for i in range(len(matrix_1))]
	for i in range(len(result)):
		for j in range(len(matrix_1[0])):
			result[i].append(matrix_1[i][j] * matrix_2[i][j])
	return result
	
#Weights matrix build
def weight_build(network_aggregate):
	place_holder = [[] for i in range(len(network_aggregate))]
	for i in range(len(place_holder)):
		for j in range(network_aggregate[i].get_layer_size()):
			place_holder[i].append(network_aggregate[i].layer[j].weights)
	return place_holder
	
#Bias vector build
def bias_build(network_aggregate):
	place_holder = [[] for i in range(len(network_aggregate))]
	for i in range(len(network_aggregate)):
		for j in range(network_aggregate[i].get_layer_size()):
			place_holder[i].append(network_aggregate[i].layer[j].bias)
	return place_holder
	
#Activation (outputs) matrix build
def activation_build(input, network_aggregate):
	place_holder = [[] for i in range(len(network_aggregate))]
	input_holder = [input]
	for i in range(len(place_holder)):
		mod_holder = network_aggregate[i].run(input_holder)
		mod_holder = mod_holder[0]
		for j in range(len(mod_holder)):
			place_holder[i].append(mod_holder[j])
		input_holder = [mod_holder]
	return place_holder
	
#Weighted inputs matrix build
def weighted_input_build(input, activation_matrix):
	place_holder = [[] for i in range(len(network_aggregate))]
	input_holder = input
	for i in range(len(place_holder)):
		for j in range(len(network_aggregate[i].layer)):
			place_holder[i].append(dot_product(input_holder, network_aggregate[i].layer[j].weights) \
				+ network_aggregate[i].layer[j].bias)
		input_holder = activation_matrix[i]
	return place_holder
	
#Partial of cost w.r.t. final output
def aL_partial_build(final_output, known_values):
	place_holder = []
	for i in range(len(final_output)):
		place_holder_1 = final_output[i][0]
		place_holder.append(place_holder_1 - known[i])
	return place_holder
	
#Weights matrix update function
def weights_update(weights_matrix, partials, input_size, layer_1_size, percent, rate):
	place_holder = weights_matrix[:][:][:]
	for i in range(len(partials)):
		if i >= (input_size * layer_1_size):
			index_1 = 1
			index_3 = i % layer_1_size
		else:
			index_1 = 0
			index_3 = i % (input_size)
		if i < (input_size * layer_1_size):
			index_2 = floor(i / input_size)
		else:
			index_2 = 0
		
		weights_matrix[index_1][index_2][index_3] = weights_matrix[index_1][index_2][index_3] \
			- ((rate / 1) * partials[i])
	
	return weights_matrix
			
#Bias vector update function
def bias_update(bias_vectors, deltas, percent, rate):
	layer_1_list = []
	#One place holder for each neuron in layer 1
	for i in range(len(deltas[0])):
		layer_1_list.append(deltas[0][i][0])
	#One place holder for each neuron in layer 2
	place_holder_10 = deltas[1][0][0]
	place_holder_1 = [layer_1_list, [place_holder_10]]
	
	for i in range(len(bias_vectors)):
		for j in range(len(bias_vectors[i])):
			bias_vectors[i][j] = bias_vectors[i][j] - (rate / 1) * place_holder_1[i][j]
	return bias_vectors

#Neuron bias attribute update using updated bias vector
def neuron_bias_update(bias_vectors, network_aggregate):
	for i in range(len(bias_vectors)):
		for j in range(len(bias_vectors[i])):
			network_aggregate[i].layer[j].bias = bias_vectors[i][j]

#Neuron weight attribute update using updated weight matrix
def neuron_weight_update(weight_matrices, network_aggregate):
	for i in range(len(network_aggregate)):
		for j in range(network_aggregate[i].get_layer_size()):
			network_aggregate[i].layer[j].weights = weight_matrices[i][j]
			
#Neuron Class
class neuron(object):
	#Initialize random weights equal in number to number of input variables
	def __init__(self, input_size):
		self.weights = [uniform(-1.0,1.0) for n in range(input_size)]
		self.bias = -1.0
	
	#Take inputs and generate output
	def feedforward(self, inputs, index):
		weighted_sum = dot_product(inputs[index], self.weights) + self.bias
		return activation(weighted_sum)
		
	#Attribute update functions
	def update_bias(self, updated_bias):
		self.bias = updated_bias
		
	def update_weights(self, updated_weights):
		self.weights = updated_weights

#Layer Class (aggregation of neurons into layers)
class network_layer(object):
	#For each neuron in layer add it to the list
	def __init__(self, *args):
		self.layer = [value for (number, value) in enumerate(args)]
			
	#Return the size of the layer
	def get_layer_size(self):
		return len(self.layer)
		
	#Create output list then run the feedforward for each neuron over each
	#data point and add output to the list
	def run(self, inputs):
		self.output = []
		for i in range(len(inputs)):
			self.output.append([])
		for neuron in self.layer:
			for i in range(len(inputs)):
				self.output[i].append(neuron.feedforward(inputs, i))
		return self.output
#******************************************************************************

#Program
#******************************************************************************
#Initialize variables
inputs = []
known = [""]
network_aggregate = [] #List of layer objects
learning_rate = 0.01
cycles = 2000

#Get input data and mark a subset for training
inputs_f = open("c:\\Users\\Anthony\\Documents\\osc_data_2.csv", 'r')
for row in inputs_f:
	row = row.split(',')
	for i in range(len(row)):
		row[i] = float(row[i])
	inputs.append(row)
inputs_f.close()
inputs_old = inputs[:]
input_subset_size = floor((8 * len(inputs)) / 10)

#Get known results data
known_f = open("c:\\Users\\Anthony\\Documents\\known.csv", 'r')
for row in known_f:
	row = row[0 : len(row) - 1]
	known.append(row)
known_f.close()
known = known[1:]

#Convert training strings to numbers
for i in range(len(known)):
	known[i] = float(known[i])

#Initialize neurons and their layers with input data,
#add each layer to combined layer list
a = neuron(len(inputs[0]))
b = neuron(len(inputs[0]))
c = neuron(len(inputs[0]))
d = neuron(len(inputs[0]))
#e = neuron(len(inputs[0]))
#f = neuron(len(inputs[0]))
#g = neuron(len(inputs[0]))
layer_1 = network_layer(a, b, c, d)
network_aggregate.append(layer_1)

zeta = neuron(layer_1.get_layer_size())
layer_2 = network_layer(zeta)
network_aggregate.append(layer_2)

#Build matrix of weight matrices for each layer
weight_matrices = weight_build(network_aggregate)
			
#Build bias vectors
bias_vectors = bias_build(network_aggregate)

for j in range(cycles):
	for i in range(input_subset_size):
		deltas = []
		#Build activation matrix
		activations = activation_build(inputs[i], network_aggregate)
		
		#Build weighted inputs matrix
		weighted_input = weighted_input_build(inputs[i], activations)
		
		#Add inputs to activation matrix
		activations.insert(0, inputs[i])
		
		#Run the network
		output_1 = layer_1.run([inputs[i]])
		output_2 = layer_2.run(output_1)

		#Partial derivative of cost w.r.t. final layer outputs
		partial_wrt_aL = aL_partial_build(output_2, known)
		
		#Calculate delta for last layer
		y = [act_derivative_to_matrix(weighted_input)[-1]]
		delta_L = hadamard_product([partial_wrt_aL], y)
		deltas.append(delta_L)
		
		#Calculate deltas for the rest of the layers
		delta_x = []
		for i in reversed(range(len(weighted_input) - 1)):
			z = [act_derivative_to_matrix(weighted_input)[i]]
			for j in range(len(z)):
				for k in range(len(z[j])):
					delta_x.append(hadamard_product(matrix_multiply(transpose([weighted_input[-1]]), \
						deltas[0]), [[z[j][k]]]))
			deltas.insert(0, delta_x)
		
		#Some reformatting of the delta data to make it easier to work with than what resulted earlier
		layer_1_list = deltas[0][0]
		for i in range(1, len(deltas[0])):
			layer_1_list.append(deltas[0][i][0])
		place_holder_10 = deltas[1]
		deltas = [layer_1_list, place_holder_10]
		
		#Calculate partials_wrt_weights as an array
		partials_wrt_weights = []
		for k in range(len(deltas)):
			for m in range(len(deltas[k])):
				for j in range(len(activations[k])):
					partials_wrt_weights.append(activations[k][j] * deltas[k][m][0])
		
		#Update the weights, and biases after running the network			
		weight_matrices = weights_update(weight_matrices, partials_wrt_weights, len(inputs[0]), \
			network_aggregate[0].get_layer_size(), input_subset_size, learning_rate)
		bias_vectors = bias_update(bias_vectors, deltas, input_subset_size, learning_rate)
		
		#Apply updated weights and biases to each neuron
		neuron_bias_update(bias_vectors, network_aggregate)
		neuron_weight_update(weight_matrices, network_aggregate)
	
	#Randomize the data
	for i in range(input_subset_size):
		rand = randint(0, input_subset_size - 1)
		inputs[rand], inputs[i] = inputs[i], inputs[rand]
		known[rand], known[i] = known[i], known[rand]

#Print results over entire input set
f = open("results.csv", 'w')
for i in range(len(inputs_old)):
	output_1 = layer_1.run([inputs_old[i]])
	output_2 = layer_2.run(output_1)
	f.write(str(output_2[0][0])+ ",")
f.close()
