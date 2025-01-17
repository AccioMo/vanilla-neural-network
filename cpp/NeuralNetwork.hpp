
#ifndef NEURALNETWORK_HPP
# define NEURALNETWORK_HPP

# include <iostream>
# include "NetworkLayer.hpp"

class NeuralNetwork {
	private:
		int		_size;
		double	_learning_rate;

	public:
		std::vector<NetworkLayer>	layers;

		/* `nodes` is an array of `size + 1`. It should contain 
		the size (aka. num of nodes) of each layer. */
		NeuralNetwork( int size, int *nodes, double learning_rate );
		NeuralNetwork( const NeuralNetwork &og );
		~NeuralNetwork( );

		double	feedforward( void ) const;
		double	backpropagation( double outputs ) const;
		void	training( double *errors, double *deltas );
};

#endif
