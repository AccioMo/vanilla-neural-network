
#include "NetworkLayer.hpp"

NetworkLayer::NetworkLayer( int input_size, int output_size )
	: _type("none"),
	_neurons(input_size),
	_weights(Matrix(input_size, output_size, \
		xavier_glorot_init(input_size, output_size))),
	_biases(Matrix(1, output_size, 0.1))
{ }

NetworkLayer::NetworkLayer( const NetworkLayer &og )
	: _type(og._type),
	_neurons(og._neurons),
	_weights(og._weights),
	_biases(og._biases)
{ }

NetworkLayer	&NetworkLayer::operator=( const NetworkLayer &og ) {
	this->_type = og._type;
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

void	NetworkLayer::update( const Matrix &prev_outputs, double learning_rate ) {
	this->_weights = this->_weights - ((prev_outputs.transpose() * this->_deltas) * learning_rate);
	this->_biases = this->_biases - (this->_deltas.sum_columns() * learning_rate);
}

void    NetworkLayer::setWeights( const Matrix &new_weights ) {
	this->_weights = new_weights;
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
