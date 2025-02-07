
#include "NetworkLayer.hpp"

NetworkLayer::NetworkLayer( int input_size, int output_size )
	: _type("none"),
	_neurons(output_size),
	_weights(Matrix(input_size, output_size, \
		xavier_glorot_init(input_size, output_size))),
	_biases(Matrix(1, output_size, 0.1)),
	_m(Matrix(input_size, output_size, 0.0)),
	_v(Matrix(input_size, output_size, 0.0))
{ }

void	NetworkLayer::update( const Matrix &inputs, 
							  double learning_rate, 
							  int timestep, 
							  double l2_reg, 
							  double beta1, 
							  double beta2 ) {
	Matrix	weight_gradient = inputs.transpose() * this->_deltas;
	weight_gradient = weight_gradient + (this->_weights * l2_reg);

	this->_m = this->_m * beta1 + weight_gradient * (1.0 - beta1);
	this->_v = this->_v * beta2 + weight_gradient.square() * (1.0 - beta2);

	Matrix	m_hat = this->_m / (1.0 - std::pow(beta1, timestep));
	Matrix	v_hat = this->_v / (1.0 - std::pow(beta2, timestep));

	this->_weights = this->_weights - (m_hat / v_hat.sqrt() + 1e-8) * learning_rate;

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
