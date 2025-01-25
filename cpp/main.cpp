
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

	// int	nodes[NETWORK_SIZE] = {IMAGE_SIZE, 128, 64, POSSIBILE_OUTPUTS};
	// NeuralNetwork	network(NETWORK_SIZE, (int *)nodes, LEARNING_RATE);

	NeuralNetwork	network("configs/60k-005-40i.bin");

	/* ------------------------------------------------------------------------------------------------ */

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

	std::vector<unsigned char> mnist_test_images = read_binary_file("mnist/mnist_test_images.bin", IMAGE_SIZE * TEST_SIZE);

	Matrix	testing_batch_input(TEST_SIZE, IMAGE_SIZE);

	for (int i = 0; i < TEST_SIZE; i++) {
		for (int j = 0; j < IMAGE_SIZE; j++) {
			testing_batch_input.m[i][j] = static_cast<double>(mnist_test_images[(i * IMAGE_SIZE) + j]);
		}
	}

	std::vector<unsigned char> mnist_test_labels = read_binary_file("mnist/mnist_test_labels.bin", TEST_SIZE);

	Matrix	testing_batch_output(TEST_SIZE, POSSIBILE_OUTPUTS);

	for (int i = 0; i < TEST_SIZE; i++) {
		for (int j = 0; j < POSSIBILE_OUTPUTS; j++) {
			testing_batch_output.m[i][static_cast<int>(mnist_test_labels[i])] = 1.0;
		}
	}

	Matrix	t_inputs = testing_batch_input.normalize(testing_batch_input.min(), testing_batch_input.max());
	Matrix	t_outputs = testing_batch_output.normalize(testing_batch_output.min(), testing_batch_output.max());

	Matrix result = network.test(t_inputs);

	std::cout << "   --- TESTING ---" << std::endl;
	std::cout << result << std::endl;
	int	choices[POSSIBILE_OUTPUTS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	for (int k = 0; k < TEST_SIZE; k++) {
		double d = 0.0;
		for (int i = 0; i < POSSIBILE_OUTPUTS; i++) {
			d += result.m[k][i];
			std::cout << choices[i] << std::fixed << std::setprecision(2);
			std::cout << " : " << result.m[k][i] * 100.0 << "% certainty";
			if (t_outputs.m[k][i] == 1)
				std::cout << " (correct)";
			std::cout << std::endl;
		}
		std::cout << "d: " << d << std::endl;
		std::cout << "	---  " << std::endl;
	}

    return (0);
}