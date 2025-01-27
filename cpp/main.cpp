
#include "NeuralNetwork.hpp"
#include "NetworkLayer.hpp"
#include "Matrix.hpp"
#include <vector>

int main( int ac, char **av )
{
	NeuralNetwork	network = NeuralNetwork("configs/60k-005-40i.bin");

	if (ac > 1)
		network.runOnImage(av[1]);
	else
		std::cerr << "usage: ./NeuralNetwork <image_path>" << std::endl;

    return (0);
}