
#ifndef OUTPUTLAYER_HPP
# define OUTPUTLAYER_HPP

# include "NetworkLayer.hpp"

class OutputLayer: public NetworkLayer {
	public:
		OutputLayer( int input_size, int output_size );
		OutputLayer( const OutputLayer &og );
		~OutputLayer() override;

		void	backpropagation( const Matrix &expected_outputs );
};

#endif
