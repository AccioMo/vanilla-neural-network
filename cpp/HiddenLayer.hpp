
#ifndef HIDDENLAYER_HPP
# define HIDDENLAYER_HPP

# include "NetworkLayer.hpp"

class HiddenLayer: public NetworkLayer {
	private:
		const int	_index;

	public:
		HiddenLayer( int index, int input_size, int output_size );
		HiddenLayer( const HiddenLayer &og );
		HiddenLayer	&operator=( const HiddenLayer &og );
		~HiddenLayer() override;

		void	backpropagation( const NetworkLayer &next_layer );
		
		int		getIndex( void ) const;
};

#endif
