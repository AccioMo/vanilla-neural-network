
#include "NetworkLayer.hpp"

NetworkLayer::NetworkLayer( int input_size, int output_size )
	: _type("none"),
	_neurons(output_size),
	_weights(Matrix(input_size, output_size, \
		xavier_glorot_init(input_size, output_size))),
	_biases(Matrix(1, output_size, 0.1))
{ }

void	NetworkLayer::update( const Matrix &inputs, double learning_rate ) {
	Matrix	weight_gradient = inputs.transpose() * this->_deltas;
	weight_gradient = weight_gradient + (L2_LAMBDA_REGULARIZATION * sum(this->_weights));
	
	this->_weights = this->_weights - (weight_gradient * learning_rate);
	this->_biases = this->_biases - (this->_deltas.sum_columns() * learning_rate);
}

NetworkLayer::~NetworkLayer() { }

Matrix  NetworkLayer::getWeights( void ) const {
    return (this->_weights);
}

void    NetworkLayer::setWeights( const Matrix &new_weights ) {
	this->_weights = new_weights;
}

int  NetworkLayer::getSize( void ) const {
    return (this->_neurons);
}

void    NetworkLayer::setSize( int new_size ) {
	this->_neurons = new_size;
}

Matrix  NetworkLayer::getBiases( void ) const {
    return (this->_biases);
}

void    NetworkLayer::setBiases( Matrix &new_biases ) {
	this->_biases = new_biases;
}

Matrix  NetworkLayer::getOutputs( void ) const {
    return (this->_outputs);
}

void    NetworkLayer::setOutputs( Matrix &new_outputs ) {
	this->_outputs = new_outputs;
}

Matrix  NetworkLayer::getErrors( void ) const {
    return (this->_errors);
}

void    NetworkLayer::setErrors( Matrix &new_errors ) {
	this->_errors = new_errors;
}

Matrix  NetworkLayer::getDeltas( void ) const {
    return (this->_deltas);
}

void    NetworkLayer::setDeltas( Matrix &new_deltas ) {
	this->_deltas = new_deltas;
}

std::string  NetworkLayer::getType( void ) const {
    return (this->_type);
}

std::ostream	&operator<<( std::ostream &os, NetworkLayer &nl ) {
	os << "\t --- " << nl.getType() << " layer --- " << std::endl;
	os << "  < Weights > " << std::endl << nl.getWeights() << std::endl;
	os << std::endl;
	os << "  < Biases > " << std::endl << nl.getBiases() << std::endl;
	return (os);
}
