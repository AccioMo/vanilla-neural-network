
#include "HiddenLayer.hpp"

HiddenLayer::HiddenLayer( int index, int input_size, int output_size )
	: NetworkLayer(index, input_size, output_size)
{ }

HiddenLayer::HiddenLayer( const HiddenLayer &og )
	: NetworkLayer(og)
{ }

HiddenLayer::~HiddenLayer() { }

void	HiddenLayer::backpropagation( const NetworkLayer &next_layer ) {
	this->_errors = next_layer.getDeltas() * next_layer.getWeights().transpose();
	this->_deltas = this->_errors * sigmoid_derivative(this->_outputs);
}
