
#include "OutputLayer.hpp"

OutputLayer::OutputLayer( int index, int input_size, int output_size )
	: NetworkLayer(index, input_size, output_size)
{ }

OutputLayer::OutputLayer( const OutputLayer &og )
	: NetworkLayer(og)
{ }

OutputLayer::~OutputLayer() { }

void	OutputLayer::backpropagation( const Matrix &expected_outputs ) {
	this->_errors = expected_outputs - this->_outputs;
	this->_deltas = this->_errors * sigmoid_derivative(expected_outputs);
}
