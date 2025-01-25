
#include "HiddenLayer.hpp"

HiddenLayer::HiddenLayer( int index, int input_size, int output_size )
	: NetworkLayer(input_size, output_size),
	_index(index)
{
	this->_type = "hidden";
}

HiddenLayer::HiddenLayer( const HiddenLayer &og )
	: NetworkLayer(og),
	_index(og._index)
{ }

HiddenLayer	&HiddenLayer::operator=( const HiddenLayer &og )
{
	if (this == &og)
		return *this;
	return *this;
}

HiddenLayer::~HiddenLayer() { }

Matrix	&HiddenLayer::feedforward( const Matrix &prev_outputs ) {
	Matrix	logit = (prev_outputs * this->_weights) + this->_biases;
	this->_outputs = ReLU(logit);
	return (this->_outputs);
}

void	HiddenLayer::backpropagation( const NetworkLayer &next_layer ) {
	this->_errors = next_layer.getDeltas() * next_layer.getWeights().transpose();
	this->_deltas = this->_errors.hadamard_product(ReLU_derivative(this->_outputs));
}

int  HiddenLayer::getIndex( void ) const {
    return (this->_index);
}
