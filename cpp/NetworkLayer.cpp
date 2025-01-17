
#include "NetworkLayer.hpp"

NetworkLayer::NetworkLayer( int input_size, int output_size ) {
	_neurons = input_size;
	_weights = Matrix(output_size, input_size);
	_biases = Matrix(output_size, 1);
}

NetworkLayer::NetworkLayer( const NetworkLayer &og ) {
	this->_neurons = og._neurons;
	this->_weights = og._weights;
	this->_biases = og._biases;
	this->_outputs = og._outputs;
	this->_errors = og._errors;
	this->_deltas = og._deltas;
}

NetworkLayer	&NetworkLayer::operator=( const NetworkLayer &og ) {
	this->_neurons = og._neurons;
	this->_weights = og._weights;
	this->_biases = og._biases;
	this->_outputs = og._outputs;
	this->_errors = og._errors;
	this->_deltas = og._deltas;
	return (*this);
}

NetworkLayer::~NetworkLayer() { }

Matrix  NetworkLayer::getWeights( void ) const {
    return (this->_weights);
}

void    NetworkLayer::setWeights( Matrix new_weights ) {
	this->_weights = new_weights;
}

Matrix  NetworkLayer::getBiases( void ) const {
    return (this->_biases);
}

void    NetworkLayer::setBiases( Matrix new_biases ) {
	this->_biases = new_biases;
}
