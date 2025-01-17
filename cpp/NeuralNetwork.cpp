
#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork( int size, int *nodes, double learning_rate )
	: _size(size),
	_learning_rate(learning_rate) {
	this->layers.reserve(size);
	for (int i = 0; i < size; i++) {
		this->layers.emplace_back(NetworkLayer(i, nodes[i], nodes[i + 1]));
	}
}

NeuralNetwork::NeuralNetwork( const NeuralNetwork &og ) {
	(void)og;
}

NeuralNetwork::~NeuralNetwork() {

}

double	NeuralNetwork::feedforward( void ) const {
	return _learning_rate + _size;
}

double	NeuralNetwork::backpropagation( double outputs ) const {
	return outputs * _size;
}

void	NeuralNetwork::training( double *errors, double *deltas ) {
	(void)errors;
	(void)deltas;
}

