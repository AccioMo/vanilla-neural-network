
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

		/* `nodes` is an array of size `size` (hehe). It should contain 
		the size - aka. num of nodes (neurons) - of each layer. */
		NeuralNetwork( int size, int *nodes, double learning_rate );
		NeuralNetwork( const NeuralNetwork &og );
		NeuralNetwork	&operator=( const NeuralNetwork &og );
		~NeuralNetwork( );

		void	feedforward( const Matrix &inputs );
		void	backpropagation( void );
		void	training( void );
};

#endif
