
#ifndef NETWORKLAYER_HPP
# define NETWORKLAYER_HPP

# include <iostream>
# include "Matrix.hpp"

class NetworkLayer {
	private:
		int	_neurons;

		/* these weights and biases should 
		connect this layer with the next one */
		Matrix		_weights;
		Matrix		_biases;

		Matrix		_outputs;
		Matrix		_errors;
		Matrix		_deltas;

	public:
		NetworkLayer( int input_size, int output_size );
		NetworkLayer( const NetworkLayer &og );
		NetworkLayer	&operator=( const NetworkLayer &og );
		~NetworkLayer();

		Matrix  getWeights( void ) const;
		void    setWeights( Matrix new_weights );
		Matrix  getBiases( void ) const;
		void    setBiases( Matrix new_biases );
};

#endif
