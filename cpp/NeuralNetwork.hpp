
#ifndef NEURALNETWORK_HPP
# define NEURALNETWORK_HPP

# include <iostream>

# include "HiddenLayer.hpp"
# include "OutputLayer.hpp"

class NeuralNetwork {
	private:
		int		_size;
		double	_learning_rate;

	public:
		OutputLayer					output_layer;
		std::vector<HiddenLayer>	hidden_layers;

		/* `nodes` is an array of size `size` (hehe). It should contain 
		the size - aka. num of nodes (neurons) - of each layer. */
		NeuralNetwork( int size, int *nodes, double learning_rate );
		NeuralNetwork( const NeuralNetwork &og );
		NeuralNetwork	&operator=( const NeuralNetwork &og );
		~NeuralNetwork( );

		void	feedforward( const Matrix &inputs );
		void	backpropagation( const Matrix &expected_outputs );
		void	training( void );
};

#endif
