
#ifndef NEURALNETWORK_HPP
# define NEURALNETWORK_HPP

# include <iostream>

class NeuralNetwork {
	private:
		double			learning_rate;
		int				num_layers;
		NetworkLayer	*layers;

	public:
		double	feedforward( void ) const;
		double	backpropagation( double outputs ) const;
		void	training( double *errors, double *deltas );
};

#endif
