#include "NeuralNetwork.hpp"
#include "NetworkLayer.hpp"
#include "Matrix.hpp"
#include <vector>

#define NETWORK_SIZE 3
#define HIDDEN_LAYERS NETWORK_SIZE - 2

#include <sys/time.h>
double	ft_get_time(void)
{
	struct timeval	counter;

	gettimeofday(&counter, NULL);
	return (counter.tv_sec * 1000.0 + counter.tv_usec / 1000.0);
}

int main( void )
{
	Matrix	raw_inputs((std::vector<std::vector<double>>){{0, 1},
													 {1, 0},
													 {0, 0},
													 {1, 1}});

	Matrix	raw_outputs((std::vector<std::vector<double>>){{1},
													   {1},
													   {0},
													   {0}});

	int	nodes[NETWORK_SIZE] = {2, 4, 1};

	Matrix	inputs = raw_inputs.normalize(0, 1);
	Matrix	outputs = raw_outputs.normalize(0, 1);

	NeuralNetwork	network(NETWORK_SIZE, (int *)nodes, 0.5);
	
	std::cout << "--- PRE TRAINING ---" << std::endl;
	std::cout << network.hidden_layers[0] << std::endl;
	std::cout << network.output_layer << std::endl;

	double start = ft_get_time();
	network.training(inputs, outputs, 50000);

	std::cout << "--- POST TRAINING ---" << std::endl;
	std::cout << network.hidden_layers[0] << std::endl;
	std::cout << network.output_layer << std::endl;
	std::cout << "errors	: " << network.output_layer.getErrors() << std::endl;
	std::cout << "outputs	: " << network.output_layer.getOutputs() << std::endl;
	std::cout << "accuracy	: " << (1 - network.output_layer.getErrors().abs().mean()) * 100 << "%" << std::endl;
	std::cout << "time		: " << (ft_get_time() - start) / 1000 << "s" << std::endl;

	// while (1) {
	// 	double	in1, in2;
	// 	std::cout << "Enter a number: ";
	// 	std::cin >> in1;
	// 	std::cout << "Enter a number: ";
	// 	std::cin >> in2;
	// 	Matrix in_matrix((std::vector<std::vector<double>>){{in1, in2}});
	// 	std::cout << in1 << " ^ " << in2 << " is " << network.test(in_matrix) << std::endl;
	// }

    return (0);
}