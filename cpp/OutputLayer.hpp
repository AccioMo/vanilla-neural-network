
#ifndef OUTPUTLAYER_HPP
# define OUTPUTLAYER_HPP

# include "NetworkLayer.hpp"

class OutputLayer: public NetworkLayer {
	public:
		OutputLayer( int input_size, int output_size );
		OutputLayer( const OutputLayer &og );
		~OutputLayer() override;

		Matrix	&feedforward( const Matrix &prev_outputs ) override;
		void	backpropagation( const Matrix &expected_outputs );
};

#endif
