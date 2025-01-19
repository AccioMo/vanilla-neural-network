
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
	(void)og;
	return (*this);
}

HiddenLayer::~HiddenLayer() { }

void	HiddenLayer::backpropagation( const NetworkLayer &next_layer ) {
	this->_errors = next_layer.getDeltas() * next_layer.getWeights().transpose();
	this->_deltas = this->_errors.hadamard_product(sigmoid_derivative(this->_outputs));
}

int  HiddenLayer::getIndex( void ) const {
    return (this->_index);
}
