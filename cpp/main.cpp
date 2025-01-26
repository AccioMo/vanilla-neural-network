
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

std::vector<Matrix>	get_input_batch( const char *filename ) {
    std::ifstream file(filename, std::ios::binary);
	std::vector<Matrix>	inputs;

	int	size = TRAIN_SIZE / BATCH_SIZE;
    if (file.is_open()) {

		for (int k = 0; k < size; k++) {
			Matrix	batch_matrix(BATCH_SIZE, IMAGE_SIZE);

			std::vector<unsigned char> data(BATCH_SIZE * IMAGE_SIZE);
			file.read(reinterpret_cast<char *>(data.data()), BATCH_SIZE * IMAGE_SIZE);

			for (int i = 0; i < BATCH_SIZE; i++) {
				for (int j = 0; j < IMAGE_SIZE; j++) {
					batch_matrix.m[i][j] = static_cast<double>(data[(i * IMAGE_SIZE) + j]);
				}
			}
			inputs.push_back(batch_matrix);
		}
		file.close();
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
	return (inputs);
}

std::vector<Matrix>	get_input_labels( const char *filename ) {
    std::ifstream file(filename, std::ios::binary);
	std::vector<Matrix>	outputs;

	int	size = TRAIN_SIZE / BATCH_SIZE;
    if (file.is_open()) {

		for (int k = 0; k < size; k++) {
			Matrix	batch_matrix(BATCH_SIZE, POSSIBILE_OUTPUTS);

			std::vector<unsigned char> data(BATCH_SIZE);
			file.read(reinterpret_cast<char *>(data.data()), BATCH_SIZE);

			for (int i = 0; i < BATCH_SIZE; i++) {
				batch_matrix.m[i][static_cast<int>(data[i])] = 1.0;
			}
			outputs.push_back(batch_matrix);
		}
		file.close();
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
	return (outputs);
}

int main( void )
{
	/*
	int	nodes[NETWORK_SIZE] = {IMAGE_SIZE, 128, 64, POSSIBILE_OUTPUTS};
	NeuralNetwork	network(NETWORK_SIZE, (int *)nodes, LEARNING_RATE);
	*/

	NeuralNetwork	network("configs/60k-005-40i.bin");

	/* --------------------------------------------- training --------------------------------------------- */

	/*
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
		std::cout << network.output_layer.getWeights() << std::endl;
		std::cout << network.output_layer.getOutputs() << std::endl;
		std::cout << "   ---	" << std::endl;
	}

	std::cout << "Training done!" << std::endl << std::endl;
	std::cout << "time		: " << (ft_get_time() - start) / 1000 << "s" << std::endl;
	std::cout << "   ---	" << std::endl;

	std::cout << std::endl;

	std::cout << "   --- TRAINING RESULTS ---" << std::endl;

	network.saveConfigJson("configs/60k-005-60i.json");
	network.saveConfigBin("configs/60k-005-60i.bin");
	*/

	/* --------------------------------------------- testing --------------------------------------------- */

	std::vector<Matrix>	t_inputs = get_input_batch("mnist/mnist_test_images.bin");

	std::vector<Matrix>	t_outputs = get_input_labels("mnist/mnist_test_labels.bin");

	std::cout << "   --- TESTING ---" << std::endl;

	int	total_iterations = TEST_SIZE / BATCH_SIZE;
	for (int i = 0; i < total_iterations; i++) {
		std::cout << "Iteration " << i + 1 << " of " << total_iterations << std::endl;
		Matrix	normalized_inputs = t_inputs[i].normalize(INPUT_MIN, INPUT_MAX);
		Matrix	normalized_outputs = t_outputs[i].normalize(OUTPUT_MIN, OUTPUT_MAX);
		network.feedforward(normalized_inputs);
		network.backpropagation(normalized_outputs);
		network.printData(normalized_outputs);
		std::cout << "   ---	" << std::endl;
	}

    return (0);
}