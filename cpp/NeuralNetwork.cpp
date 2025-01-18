
#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork( int size, int *nodes, double learning_rate )
	: _size(size),
	_learning_rate(learning_rate),
	output_layer(OutputLayer(size - 2, nodes[size - 2], nodes[size - 1])) {
	this->hidden_layers.reserve(size - 2);
	for (int i = 0; i < size - 2; i++) {
		this->hidden_layers.emplace_back(HiddenLayer(i, nodes[i], nodes[i + 1]));
	}
}

NeuralNetwork::NeuralNetwork( const NeuralNetwork &og )
	: _size(og._size),
	_learning_rate(og._learning_rate),
	output_layer(og.output_layer),
	hidden_layers(og.hidden_layers)
{ }

NeuralNetwork	&NeuralNetwork::operator=( const NeuralNetwork &og ) {
	this->_size = og._size;
	this->_learning_rate = og._learning_rate;
	this->output_layer = og.output_layer;
	this->hidden_layers = og.hidden_layers;
	return (*this);
}

NeuralNetwork::~NeuralNetwork() { }

void	NeuralNetwork::feedforward( const Matrix &inputs ) {
	Matrix	outputs = inputs;
	for (auto &layer : hidden_layers) {
		outputs = layer.feedforward(outputs);
	}
	output_layer.feedforward(outputs);
}

void	NeuralNetwork::backpropagation( const Matrix &expected_outputs ) {
	output_layer.backpropagation(expected_outputs);
	NetworkLayer	*prev_layer = &output_layer;
	for (int i = hidden_layers.size() - 1; i >= 0; i--) {
		this->hidden_layers[i].backpropagation(*prev_layer);
		prev_layer = &this->hidden_layers[i];
	}
}

void	NeuralNetwork::training( void ) {
}
