
#ifndef NEURALNETWORK_HPP
# define NEURALNETWORK_HPP

# include <iostream>
# include <fstream>
# include <cstring>

# include "HiddenLayer.hpp"
# include "OutputLayer.hpp"
# include "Matrix.hpp"
# include "config.hpp"
# include "math.hpp"
# include "utils.hpp"

class NeuralNetwork {
	private:
		int		_size;
		double	_learning_rate;
		Matrix	_entropy;
		Matrix	_confidence;

		Matrix	_beta1;
		Matrix	_beta2;

	public:
		OutputLayer					output_layer;
		std::vector<HiddenLayer>	hidden_layers;

		/* `nodes` is an array of size `size` (hehe). It should contain 
		the size - aka. num of nodes (neurons) - of each layer. */
		NeuralNetwork( );
		NeuralNetwork( int size, int *nodes );
		NeuralNetwork( const char *filename );
		~NeuralNetwork( );

		/* ... feedforward ... */
		void	feedforward( const Matrix &inputs );

		/* ... backpropagation ... */
		void	backpropagation( const Matrix &expected_outputs );

		/* ... update ... */
		void	update( const Matrix &inputs );

		/* ... training ... */
		void	train( Matrix input_batch, Matrix output_batch, int epochs );
		void	trainOnFile( const char *filename, const char *labels, const char *output_file );

		/* ... testing ... */
		void	test( const Matrix input, const Matrix expected_outputs );
		void	testOnFile( const char *filename, const char *labels );

		Matrix	run( const Matrix input );
		Matrix	runOnImage( const char *filename );

		/* ... printing ... */
		void	printData( const Matrix expected_outputs ) const;

		/* ... saving ... */
		void	saveConfigJson(const char *filename) const;
		void	saveConfigBin(const char *filename) const;

		/* ... evaluation ... */
		Matrix	calculateEntropy( void ) const;
		Matrix	calculateAccuracy( const Matrix &expected_ouputs ) const;

		/* ... getters ... */
		Matrix	getEntropy( void ) const;
		Matrix	getConfidence( void ) const;
		double	getLearningRate( void ) const;

		/* ... setters ... */
		void	setEntropy( Matrix entropy );
		void	setConfidence( Matrix confidence );
		void	setLearningRate( double learning_rate );
};

#endif
