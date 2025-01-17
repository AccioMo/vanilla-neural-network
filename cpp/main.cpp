
// #include "NeuralNetwork.hpp"
#include "NetworkLayer.hpp"
#include "Matrix.hpp"
#include <vector>

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

	NetworkLayer	hidden_layer(2, 4);
	NetworkLayer	output_layer(4, 1);

	std::cout << hidden_layer.getWeights() << std::endl;
	std::cout << output_layer.getWeights() << std::endl;

    return (0);
}