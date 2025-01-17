
#include "NeuralNetwork.hpp"
#include "NetworkLayer.hpp"
#include "Matrix.hpp"
#include <vector>

#define NETWORK_SIZE 3

int main( void )
{
	Matrix	inputs((std::vector<std::vector<double>>){{0, 1},
													 {1, 0},
													 {0, 0},
													 {1, 1}});

	Matrix	outputs((std::vector<std::vector<double>>){{1},
													   {1},
													   {0},
													   {0}});

	NetworkLayer	hidden_layer(0, 2, 4);
	NetworkLayer	output_layer(1, 4, 1);

	// std::cout << hidden_layer.getWeights() << std::endl;
	// std::cout << output_layer.getWeights() << std::endl;

	int	nodes[NETWORK_SIZE] = {2, 4, 1};
	NeuralNetwork	network(NETWORK_SIZE, (int *)nodes, 0.5);

	for (int i = 0; i < NETWORK_SIZE - 1; i++)
		std::cout << network.layers[i] << std::endl;

    return (0);
}