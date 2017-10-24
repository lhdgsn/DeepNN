// deepnet.h
// Basic deep neural net class, with options including: variable structure, common weight initialization options, regularization, ... 
// Liam Hodgson
// 23 October 2017

#ifndef _DEEPNET_H_
#define _DEEPNET_H_

class DeepNet {
private:
	double* weights; // array of pointers to arrays of weights
	double* bias; // array of bias values 
	double* X, Y; // arrays of training data
	int nFeatures;
	int nLayers;
	int* nNodes;
	initWeights();

	forwardProp(int miniBatchSize);

	BackProp();

public:
	// neural net class constructor
	// nLayers: number of layers (not including input and output layers)
	// nNodes: array containing the number of nodes in each layer
	DeepNet(int nLayers, int* nNodes, string activation);

	loadData(double* X, double* Y)

	int setVariable(string varName, double val);

	// Destructor
	~DeepNet();

	// train network
	train();
};
#endif

DeepNet::DeepNet(int nLayers, int* nNodes, string activation){
	this->nLayers = nLayers;
	this->nNodes = nNodes;
	// create all variables of appropriate size
	weights = new double[nLayers];
	for (int i = 0; i < nLayers)
		weights[i] = new double[nNodes[i]][nFeatures];



}

DeepNet::forwardProp(){
	// iterate through layers
	double Z;
	double* A = this->X;
	for(int l = 0; l < nLayers; l++)
		Z = inner_product(start(weights[l]), end(weights[l], start(X))
}