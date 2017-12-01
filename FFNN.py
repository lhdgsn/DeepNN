# FFNN.py
# functions for basic feedforward neural network
# Liam Hodgson
# 30 November 2017

import numpy as np

# load data
train_x, train_y
test_x, test_y

# list of all network weights and biases
W = []
b = []
cache = []

### define structure
# layers: list containing number of nodes in each layer (not including output layer)
# dimX, dimY: size of features and labels
def init_graph(W, b, layers, nIn, nOut):
	# init weights for each layer
	# uses xavier initialization
	# todo: change initialization based on activation
	W.append( np.random.randn((layers[0], nIn))*np.sqrt(2/nIn) )
	b.append( np.zeros((layers[0], 1)) )
	for i in range(1 : (len(layers) - 1) ):
		W.append( np.random.randn((layers[i+1], layers[i]))*np.sqrt(2/layers[i]) )
		b.append( np.zeros((layers[i+1], 1)) )
	W.append( np.random.randn((nOut, layers[-1]))*np.sqrt(2/layers[-]) )
	b.append( np.zeros((nOut, 1)) )

### forward propagation
# X: feature vector
# W, b: list of weight matrices and bias vectors
# activation: list of activations for each layer
def forward_prop(X, W, b, cache, activation):
	assert(len(activation) == len(W))

	A = X
	for l in range(len(W)):
		Z = np.dot(W[l],A) + b[l]
		if(activation[l] == 'relu'):
			A = relu(Z)
		elif(activation[l] == 'sigmoid'):
			A = sigmoid(Z)
		elif(activation[l] == 'softmax'):
			A = softmax(Z)
		else:
			raise ValueError('Invalid activation function')

	return A

# compute cost J
def cost(A, Y, type):
	assert(A.shape == Y.shape)

	if(type == 'cross_entropy'):
		tmp = np.dot(Y.T, np.log(A)) + np.dot((1-Y).T, np.log(1-A))
		cost = -np.mean( np.sum(tmp) )
	else:
		raise ValueError('Invalid cost type')

	return cost


# backward propagation
def back_prop(X, Y, W, b, activation, cost, cost_type, opti_type, cache):
	if(cost_type == 'cross_entropy'):
		assert(activation[-1] == 'sigmoid')

		## last layer
		A = cache(...) # result of output layer 
		dJdA = (1/m)*np.divide(Y-A, np.multiply(A, 1-A))
		dAdZ = np.multiply(A, 1-A)
		dJdZ = np.multiply(dJdA, dAdZ)
		# weight gradient
		dZdW = cache(...) # activation of previous layer
		dJdW = np.dot(dJdZ, dZdW.T)
		# bias gradient
		dZdb = 1
		dJdb = np.sum(dJdZ*dZdb, axis=1, keepdims=True)
		# for next layer
		dJdA_prev = W[l].T*dJdZ

		## second last layer
		# dJdW = [dJdA.dAdZ.dZdA_prev].dA_prevdZ.dZdW
		# dJdb = [dJdA.dAdZ.dZdA_prev].dA_prevdZ.dZdb
		# weight gradient
		dZdW = cache(...) # output of previous layer
		dJdW = np.dot(np.multiply(dJdA_prev, dAdZ), dZdW)
		# bias gradient
		dZdb = 1
		dJdb = np.sum(np.multiply(dJdA_prev, dAdZ), axis=1, keepdims=True)


		# # update weights and biases for each layer
		# for l in range(len(W),-1,-1):
		# 	# weight gradient
		# 	# dJ/dW = (dJ/dA)(dA/dZ)(dZ/dW=A_prev)
		# 	if(l=0):
		# 		A_prev = X
		# 	else:
		# 		A_prev = cache(A somehow)
		# 	dW = (1/m)*np.dot(dZ, A_prev.T)

		# 	# bias gradient
		# 	db = (1/m)*np.sum(dZ, axis=1, keepdims=True)

		# 	dZ = 

