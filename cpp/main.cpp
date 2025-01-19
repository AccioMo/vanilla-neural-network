
#include "NeuralNetwork.hpp"
#include "NetworkLayer.hpp"
#include "Matrix.hpp"
#include <vector>

#define NETWORK_SIZE 3
#define HIDDEN_LAYERS 1

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

	int	nodes[NETWORK_SIZE] = {2, 4, 1};
	NeuralNetwork	network(NETWORK_SIZE, (int *)nodes, 0.5);

	Matrix w1((std::vector<std::vector<double>>){{-0.162655, -1.56787, -1.33024, -0.312449}, {2.89588, 2.54925, 0.164439, 2.51641}});
	Matrix b1((std::vector<std::vector<double>>){{2.37707, -2.13075, -2.11557, 0.129395}});

	Matrix w2((std::vector<std::vector<double>>){{-1.90338}, {-0.597504}, {1.61092}, {1.86572}});
	Matrix b2((std::vector<std::vector<double>>){{-0.991096}});

	network.hidden_layers[0].setWeights(w1);
	network.hidden_layers[0].setBiases(b1);
	
	network.output_layer.setWeights(w2);
	network.output_layer.setBiases(b2);
	
	network.training(inputs, outputs, 5000);

	std::cout << network.hidden_layers[0] << std::endl;
	std::cout << network.output_layer << std::endl;

    return (0);
}