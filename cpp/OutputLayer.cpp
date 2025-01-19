
#include "OutputLayer.hpp"

OutputLayer::OutputLayer( int input_size, int output_size )
	: NetworkLayer(input_size, output_size)
{
	_type = "output";
}

OutputLayer::OutputLayer( const OutputLayer &og )
	: NetworkLayer(og)
{ }

OutputLayer::~OutputLayer() { }

void	OutputLayer::backpropagation( const Matrix &expected_outputs ) {
	this->_errors = expected_outputs - this->_outputs;
	this->_deltas = this->_errors.hadamard_product(sigmoid_derivative(this->_outputs));
}
