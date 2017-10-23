// deepnet.h
// Basic deep neural net class, with options including: variable structure, common weight initialization options, regularization, ... 
// Liam Hodgson
// 23 October 2017

#ifndef _DEEPNET_H_
#define _DEEPNET_H_

class DeepNet {
private:
	forwardProp();

	BackProp();

public:
	DeepNet();

	setVariable(string varName, double val);

	// Destructor
	~DeepNet();

	// train network
	train();


};

#endif