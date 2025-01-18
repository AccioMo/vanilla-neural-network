
#ifndef NETWORKLAYER_HPP
# define NETWORKLAYER_HPP

# include <iostream>
# include "Matrix.hpp"
# include "math.hpp"

class NetworkLayer {
	protected:
		const int	_index;
		int			_neurons;

		/* these weights and biases should 
		connect this layer with the next one */
		Matrix		_weights;
		Matrix		_biases;

		Matrix		_outputs;
		Matrix		_errors;
		Matrix		_deltas;

	public:
		NetworkLayer( int index, int input_size, int output_size );
		NetworkLayer( const NetworkLayer &og );
		NetworkLayer	&operator=( const NetworkLayer &og );
		virtual	~NetworkLayer() = 0;

		Matrix	&feedforward( const Matrix &inputs );

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
		
		int		getIndex( void ) const;
};

std::ostream	&operator<<( std::ostream &os, NetworkLayer &nl );

#endif
