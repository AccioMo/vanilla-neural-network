
#ifndef NETWORKLAYER_HPP
# define NETWORKLAYER_HPP

# include <iostream>
# include "Matrix.hpp"
# include "math.hpp"

class NetworkLayer {
	protected:
		/* type of the layer (output or hidden) */
		std::string	_type;

		/* number of neuron/nodes/units in the network */
		int			_neurons;

		/* these weights and biases should 
		connect this layer with the next one */
		Matrix		_weights;
		Matrix		_biases;

		/* predicted outputs based on this 
		layer's weights and biases */
		Matrix		_outputs;

		/* errors is the raw difference between 
		predicted and expected output, calculated 
		differenly in output and hidden layers 
		(technically should use loss function, 
		but took a shortcut hehe). 

		Here it's the intermediate state from 
		output to the 'real' back-propagated error */
		Matrix		_errors;

		/* final back-propagated error for the layer.
		calculated using the derivative of the activation 
		function and the raw error value `_errors` */
		Matrix		_deltas;

	public:
		NetworkLayer( int input_size, int output_size );
		NetworkLayer( const NetworkLayer &og );
		NetworkLayer	&operator=( const NetworkLayer &og );
		virtual	~NetworkLayer() = 0;

		/* feeds output of the previous layer 
		to the current one to predict output */
		Matrix	&feedforward( const Matrix &prev_outputs );

		/* backpropagation function in derived 
		HiddenLayer and OutputLayer classes */

		/* updates weights and biases using 
		delta from backpropagation function */
		void	update( const Matrix &prev_outputs, double learning_rate );

		Matrix  getWeights( void ) const;
		void    setWeights( const Matrix &new_weights );
		Matrix  getBiases( void ) const;
		void    setBiases( Matrix &new_biases );

		Matrix  getOutputs( void ) const;
		void    setOutputs( Matrix &new_outputs );
		Matrix  getErrors( void ) const;
		void    setErrors( Matrix &new_errors );
		Matrix  getDeltas( void ) const;
		void    setDeltas( Matrix &new_deltas );

		std::string	getType() const;
};

std::ostream	&operator<<( std::ostream &os, NetworkLayer &nl );

#endif
