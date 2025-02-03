
#ifndef HIDDENLAYER_HPP
# define HIDDENLAYER_HPP

# include "NetworkLayer.hpp"

class HiddenLayer: public NetworkLayer {
	private:
		const int	_index;

	public:
		HiddenLayer( void );
		HiddenLayer( int index, int input_size, int output_size );
		~HiddenLayer() override;

		Matrix	&feedforward( const Matrix &prev_outputs ) override;
		void	backpropagation( const NetworkLayer &next_layer );
		
		int		getIndex( void ) const;
};

#endif
