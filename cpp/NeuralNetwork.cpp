
#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork( int size, int *nodes, double learning_rate )
	: _size(size),
	_learning_rate(learning_rate) {
	this->layers.reserve(size - 1);
	for (int i = 0; i < size - 1; i++) {
		this->layers.emplace_back(NetworkLayer(i, nodes[i], nodes[i + 1]));
	}
}

NeuralNetwork::NeuralNetwork( const NeuralNetwork &og )
	: _size(og._size),
	_learning_rate(og._learning_rate),
	layers(og.layers)
{ }

NeuralNetwork	&NeuralNetwork::operator=( const NeuralNetwork &og ) {
	this->_size = og._size;
	this->_learning_rate = og._learning_rate;
	this->layers = og.layers;
	return (*this);
}

NeuralNetwork::~NeuralNetwork() { }

void	NeuralNetwork::feedforward( const Matrix &inputs ) {
	Matrix	outputs = inputs;
	for (auto &layer : layers) {
		outputs = layer.feedforward(outputs);
	}
}

void	NeuralNetwork::backpropagation( void ) {
}

void	NeuralNetwork::training( void ) {
}
