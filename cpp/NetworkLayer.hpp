
#ifndef NETWORKLAYER_HPP
# define NETWORKLAYER_HPP

# include <iostream>

class NetworkLayer {
	private:
		int		neurons;

		double  **weights;
		double  *biases;

		double	*outputs;
		double	*errors;
		double	*deltas;

	public:
		double  getWeights( void ) const;
		void    setWeights( double new_weights );
		double  getBiases( void ) const;
		void    setBiases( double new_biases );
};

#endif
