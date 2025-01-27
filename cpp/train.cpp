	
#include "NeuralNetwork.hpp"
#include "NetworkLayer.hpp"
#include "Matrix.hpp"
#include "config.hpp"
#include <fstream>
#include <vector>

#include <sys/time.h>
double	ft_get_time(void)
{
	struct timeval	counter;

	gettimeofday(&counter, NULL);
	return (counter.tv_sec * 1000.0 + counter.tv_usec / 1000.0);
}

int main( void )
{	
	/* --------------------------------------------- training --------------------------------------------- */
	
	int	nodes[NETWORK_SIZE] = {IMAGE_SIZE, 1, POSSIBILE_OUTPUTS};
	NeuralNetwork	network(NETWORK_SIZE, (int *)nodes, LEARNING_RATE);

	std::cout << "Allocating space for network..." << std::endl;

	std::vector<Matrix>	inputs = get_input_batch("mnist/mnist_train_images.bin");

	std::vector<Matrix>	outputs = get_input_labels("mnist/mnist_train_labels.bin");

	std::cout << "Network constructed!" << std::endl;

	double start = ft_get_time();
	std::cout << std::endl << "   --- TRAINING ---	" << std::endl;
	int	total_iterations = TRAIN_SIZE / BATCH_SIZE;
	for (int i = 0; i < total_iterations; i++) {
		std::cout << "Iteration " << i + 1 << " of " << total_iterations << std::endl;
		Matrix	normalized_inputs = inputs[i].normalize(INPUT_MIN, INPUT_MAX);
		Matrix	normalized_outputs = outputs[i].normalize(OUTPUT_MIN, OUTPUT_MAX);
		network.training(normalized_inputs, normalized_outputs, EPOCHS);
		network.printData(normalized_outputs);
		std::cout << "   ---	" << std::endl;
	}

	std::cout << "Training done!" << std::endl << std::endl;
	std::cout << "time		: " << (ft_get_time() - start) / 1000 << "s" << std::endl;
	std::cout << "   ---	" << std::endl;

	std::cout << std::endl;

	std::cout << "   --- TRAINING RESULTS ---" << std::endl;

	network.saveConfigJson("configs/60k-005-5i.json");
	network.saveConfigBin("configs/60k-005-5i.bin");
}
