
#include "NeuralNetwork.hpp"
#include "NetworkLayer.hpp"
#include "Matrix.hpp"
#include <vector>

#define NETWORK_SIZE 3
#define HIDDEN_LAYERS 1

#include <sys/time.h>
double	ft_get_time(void)
{
	struct timeval	counter;

	gettimeofday(&counter, NULL);
	return (counter.tv_sec * 1000.0 + counter.tv_usec / 1000.0);
}

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
	
	double start = ft_get_time();
	network.training(inputs, outputs, 5000);

	std::cout << network.hidden_layers[0] << std::endl;
	std::cout << network.output_layer << std::endl;
	std::cout << "error: \n" << network.output_layer.getErrors() << std::endl;
	std::cout << "took " << ft_get_time() - start << "ms" << std::endl;

    return (0);
}