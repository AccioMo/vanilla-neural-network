
#include "NeuralNetwork.hpp"
#include "NetworkLayer.hpp"
#include "Matrix.hpp"
#include <vector>
#include <iomanip>

#define NETWORK_SIZE 3
#define HIDDEN_LAYERS NETWORK_SIZE - 2

typedef std::vector<std::vector<double>> vec;

#include <sys/time.h>
double	ft_get_time(void)
{
	struct timeval	counter;

	gettimeofday(&counter, NULL);
	return (counter.tv_sec * 1000.0 + counter.tv_usec / 1000.0);
}

int main( void )
{
	// Matrix	raw_inputs((std::vector<std::vector<double>>){{0, 1},
	// 												 {1, 0},
	// 												 {0, 0},
	// 												 {1, 1}});

	// Matrix	raw_outputs((std::vector<std::vector<double>>){{1},
	// 												   {1},
	// 												   {0},
	// 												   {0}});
													   
	Matrix	raw_inputs((vec)
													{
													{0, 0, 0, 0,
													 0, 0, 0, 0,
													 1, 1, 1, 1,
													 0, 0, 0, 0},

													{0, 1, 0, 0,
													 0, 1, 0, 0,
													 0, 1, 0, 0,
													 0, 1, 0, 0},

													{0, 1, 0, 0,
													 0, 1, 0, 0,
													 1, 1, 1, 1,
													 0, 1, 0, 0},

													{0, 0, 0, 0,
													 1, 1, 1, 1,
													 0, 0, 0, 0,
													 0, 0, 0, 0},

													{0, 0, 0, 1,
													 0, 0, 0, 1,
													 0, 0, 0, 1,
													 0, 0, 0, 1},

													{0, 0, 1, 0,
													 1, 1, 1, 1,
													 0, 0, 1, 0,
													 0, 0, 1, 0},
													});

	Matrix	raw_outputs((std::vector<std::vector<double>>)
													{
													{0},
													{1},
													{2},
													{0},
													{1},
													{2},
													});

	int	nodes[NETWORK_SIZE] = {16, 4, 1};

	Matrix	inputs = raw_inputs.normalize(raw_inputs.min(), raw_inputs.max());
	Matrix	outputs = raw_outputs.normalize(raw_outputs.min(), raw_outputs.max());

	NeuralNetwork	network(NETWORK_SIZE, (int *)nodes, 0.5);

	Matrix weights_1((std::vector<std::vector<double>>)
		{{-0.269704, -0.772879, 1.52342, 0.603172}, 
		{-1.97492, -1.35512, -1.11777, 1.57711}, 
		{-0.727129, 1.46499, -1.80925, 2.22695}, 
		{-1.07905, -0.842372, 1.78229, 2.4908}, 
		{-0.923622, -0.164771, 1.01456, -3.00664}, 
		{-0.512091, -2.29379, 0.558249, 2.89451}, 
		{0.543507, 1.59866, 1.12665, -0.690984}, 
		{0.670191, -0.0248687, 1.07034, 1.15029}, 
		{0.0785867, -1.86295, -3.48586, -1.45357}, 
		{0.930669, -1.76189, 3.21533, -0.271558}, 
		{-2.31644, 2.89888, 0.159754, 3.09752}, 
		{-0.750095, -1.98474, -1.45866, -1.76717}, 
		{-1.12981, 2.1733, 0.972956, 2.04882}, 
		{-1.68382, -1.59807, -2.97816, 1.67032}, 
		{-0.557954, 1.13529, 1.09647, -0.19638}, 
		{0.792617, 2.53033, 1.77055, 2.34044}});


	Matrix biases_1((std::vector<std::vector<double>>)
		{{-1.70381, -1.17654, 2.82959, -2.18989}});

	Matrix weights_2((std::vector<std::vector<double>>)
		{{0.078853}, 
		{6.20097}, 
		{-6.28246}, 
		{7.4087}});

	Matrix biases_2((std::vector<std::vector<double>>)
		{{-1.41931}});
	
	network.hidden_layers[0].setWeights(weights_1);
	network.hidden_layers[0].setBiases(biases_1);

	network.output_layer.setWeights(weights_2);
	network.output_layer.setBiases(biases_2);

	// std::cout << "--- PRE TRAINING ---" << std::endl;
	// std::cout << network.hidden_layers[0] << std::endl;
	// std::cout << network.output_layer << std::endl;

	// double start = ft_get_time();
	// // network.training(inputs, outputs, 50000);

	// std::cout << "--- POST TRAINING ---" << std::endl;
	// std::cout << network.hidden_layers[0] << std::endl;
	// std::cout << network.output_layer << std::endl;
	// std::cout << "errors	: " << network.output_layer.getErrors() << std::endl;
	// std::cout << "outputs	: " << network.output_layer.getOutputs() << std::endl;
	// std::cout << "accuracy	: " << (1 - network.hidden_layers[0].getErrors().abs().mean()) * 100 << "%" << std::endl;
	// std::cout << "accuracy	: " << (1 - network.output_layer.getErrors().abs().mean()) * 100 << "%" << std::endl;
	// std::cout << "time		: " << (ft_get_time() - start) / 1000 << "s" << std::endl;

Matrix	test_inputs((std::vector<std::vector<double>>)
													{
													{0, 0, 0, 0,
													 0, 0, 1, 0,
													 0, 0, 1, 0,
													 0, 0, 0, 0}});

	Matrix result = network.test(test_inputs);

	Matrix	output_options((vec){{0.0, 1.0, 2.0}});
	output_options = output_options.normalize(output_options.min(), output_options.max());

	for (int i = 0; i < (int)output_options.m[0].size(); i++) {
		result = result.denormalize(raw_outputs.min(), raw_outputs.max());
		network.backpropagation((vec){{output_options.m[0][i]}});
		double	loss = network.output_layer.getErrors().abs().mean();
		if (i == 0)
			std::cout << "horizontal with " << std::fixed << std::setprecision(2) << (1 - loss) * 100 << "% certainty" << std::endl;
		else if (i == 1)
			std::cout << "vertical with " << std::fixed << std::setprecision(2) << (1 - loss) * 100 << "% certainty" << std::endl;
		else if (i == 2)
			std::cout << "cross with " << std::fixed << std::setprecision(2) << (1 - loss) * 100 << "% certainty" << std::endl;
	}


	// while (1) {
	// 	double	in1, in2;
	// 	std::cout << "Enter a number: ";
	// 	std::cin >> in1;
	// 	std::cout << "Enter a number: ";
	// 	std::cin >> in2;
	// 	Matrix in_matrix((std::vector<std::vector<double>>){{in1, in2}});
	// 	std::cout << network.test(in_matrix) << std::endl;
	// }

    return (0);
}