
#include "NetworkLayer.hpp"

NetworkLayer::NetworkLayer( int index, int input_size, int output_size )
	: _index(index),
	_neurons(input_size),
	_weights(Matrix(output_size, input_size, M_RAND)),
	_biases(Matrix(output_size, 1, M_RAND)),

	_outputs(Matrix(output_size, input_size)),
	_errors(Matrix(output_size, input_size)),
	_deltas(Matrix(output_size, input_size))
{ }

NetworkLayer::NetworkLayer( const NetworkLayer &og )
	: _index(og._index),
	_neurons(og._neurons),
	_weights(og._weights),
	_biases(og._biases),

	_outputs(og._outputs),
	_errors(og._errors),
	_deltas(og._deltas)
{ }

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

int  NetworkLayer::getIndex( void ) const {
    return (this->_index);
}

std::ostream	&operator<<( std::ostream &os, NetworkLayer &nl ) {
	os << "\t --- Layer " << nl.getIndex() << " --- " << std::endl;
	os << "  < Weights > " << std::endl << nl.getWeights() << std::endl;
	os << std::endl;
	os << "  < Biases > " << std::endl << nl.getBiases() << std::endl;
	return (os);
}
