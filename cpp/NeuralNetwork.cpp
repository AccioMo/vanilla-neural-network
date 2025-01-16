
#include "NeuralNetwork.hpp"

double  getWeights( void ) const {
    return (this->weights);
}

void    setWeights( double new_weights ) {
	this->weights = new_weights;
}

double  getBiases( void ) const {
    return (this->biases);
}

void    setBiases( double new_biases ) {
	this->biases = new_biases;
}


double	feedforward( void ) const {
	double	hidden_outputs = 
}

double	backpropagation( double outputs ) const {

}

void	training( double *errors, double *deltas ) {

}

