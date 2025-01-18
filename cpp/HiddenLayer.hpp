
#ifndef HIDDENLAYER_HPP
# define HIDDENLAYER_HPP

# include "NetworkLayer.hpp"

class HiddenLayer: public NetworkLayer {
	public:
		HiddenLayer( int index, int input_size, int output_size );
		HiddenLayer( const HiddenLayer &og );
		~HiddenLayer() override;

		void	backpropagation( const NetworkLayer &next_layer );
};

#endif
