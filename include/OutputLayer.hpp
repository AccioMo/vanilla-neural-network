
#ifndef OUTPUTLAYER_HPP
# define OUTPUTLAYER_HPP

# include "NetworkLayer.hpp"

class OutputLayer: public NetworkLayer {
	private:
		double	_accuracy;

	public:
		OutputLayer( void );
		OutputLayer( int input_size, int output_size );
		~OutputLayer() override;

		Matrix	&feedforward( const Matrix &prev_outputs ) override;
		void	backpropagation( const Matrix &expected_outputs );

		double	getAccuracy( void ) const;
};

#endif
