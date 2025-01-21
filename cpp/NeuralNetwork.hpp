
#ifndef NEURALNETWORK_HPP
# define NEURALNETWORK_HPP

# include <iostream>
# include <fstream>

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

		/* ... feedforward ... */
		void	feedforward( const Matrix &inputs );

		/* ... backpropagation ... */
		void	backpropagation( const Matrix &expected_outputs );
		void	update( const Matrix &inputs );

		/* ... training ... */
		void	training( Matrix input_batch, Matrix output_batch, int epochs );

		/* ... testing ... */
		Matrix	test( const Matrix input );

		/* ... saving ... */
		void	saveConfig(const char *filename) const;
};

#endif
