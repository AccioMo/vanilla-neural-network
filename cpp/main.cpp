
#include "NeuralNetwork.hpp"
#include "NetworkLayer.hpp"
#include "Matrix.hpp"
#include <vector>
#include <iomanip>
#include <fstream>

const int NETWORK_SIZE = 3;
// const int HIDDEN_LAYERS = NETWORK_SIZE - 2;

const int TRAIN_SIZE = 2000;
const int TEST_SIZE = 10;
const int IMAGE_SIZE = 28 * 28;
const int POSSIBILE_OUTPUTS = 10;

#include <sys/time.h>
double	ft_get_time(void)
{
	struct timeval	counter;

	gettimeofday(&counter, NULL);
	return (counter.tv_sec * 1000.0 + counter.tv_usec / 1000.0);
}

std::vector<unsigned char> read_binary_file(const char *filename, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    
    if (file.is_open()) {
   		std::vector<unsigned char> data(size);
        file.read(reinterpret_cast<char *>(data.data()), size);
        file.close();
	    return data;
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }

	std::vector<unsigned char> none;
    return none;
}

int main( void )
{
	std::cout << "Allocating space for network..." << std::endl;

	std::vector<unsigned char> mnist_train_images = read_binary_file("minst/mnist_train_images.bin", IMAGE_SIZE * TRAIN_SIZE);

	Matrix	training_batch_input(TRAIN_SIZE, IMAGE_SIZE);

	for (int i = 0; i < TRAIN_SIZE; i++) {
		for (int j = 0; j < IMAGE_SIZE; j++) {
			training_batch_input.m[i][j] = static_cast<double>(mnist_train_images[(i * IMAGE_SIZE) + j]);
		}
	}

	std::vector<unsigned char> mnist_train_labels = read_binary_file("minst/mnist_train_labels.bin", TRAIN_SIZE);

	Matrix	training_batch_output(TRAIN_SIZE, POSSIBILE_OUTPUTS);

	for (int i = 0; i < TRAIN_SIZE; i++) {
		for (int j = 0; j < POSSIBILE_OUTPUTS; j++) {
			training_batch_output.m[i][static_cast<int>(mnist_train_labels[i])] = 1.0;
		}
	}

	Matrix	inputs = training_batch_input.normalize(training_batch_input.min(), training_batch_input.max());
	Matrix	outputs = training_batch_output.normalize(training_batch_output.min(), training_batch_output.max());

	int	nodes[NETWORK_SIZE] = {IMAGE_SIZE, 64, POSSIBILE_OUTPUTS};

	NeuralNetwork	network(NETWORK_SIZE, (int *)nodes, 0.0001);

	std::cout << "Network constructed!" << std::endl;

	// std::cout << "--- PRE TRAINING ---" << std::endl;
	// std::cout << network.hidden_layers[0] << std::endl;
	// std::cout << network.output_layer << std::endl;

	double start = ft_get_time();
	// std::cout << "inputs: " << inputs << std::endl;
	// std::cout << "outputs: " << outputs << std::endl;
	// std::cout << "weights: " << network.output_layer.getWeights() << std::endl;
	network.training(inputs, outputs, 100);

	// std::cout << "--- POST TRAINING ---" << std::endl;
	// std::cout << network.hidden_layers[0] << std::endl;
	// std::cout << network.output_layer << std::endl;
	// std::cout << "errors	: " << network.output_layer.getErrors() << std::endl;
	// std::cout << "outputs	: " << network.output_layer.getOutputs() << std::endl;
	double max_entropy = -std::log(1.0 / (double)POSSIBILE_OUTPUTS);
	std::cout << "entropy		: " << network.calculateEntropy().mean() << " (max " << max_entropy << ")" << std::endl;
	std::cout << "accuracy	: " << network.calculateAccuracy(outputs).mean() * 100 << "%" << std::endl;
	std::cout << "confidence	: " << (1.0 - (network.calculateEntropy().mean() / max_entropy)) * 100 << "%" << std::endl;
	std::cout << "time		: " << (ft_get_time() - start) / 1000 << "s" << std::endl;


	std::vector<unsigned char> mnist_test_images = read_binary_file("minst/mnist_test_images.bin", IMAGE_SIZE * TEST_SIZE);

	Matrix	testing_batch_input(TEST_SIZE, IMAGE_SIZE);

	for (int i = 0; i < TEST_SIZE; i++) {
		for (int j = 0; j < IMAGE_SIZE; j++) {
			testing_batch_input.m[i][j] = static_cast<double>(mnist_test_images[(i * IMAGE_SIZE) + j]);
		}
	}

	std::vector<unsigned char> mnist_test_labels = read_binary_file("minst/mnist_test_labels.bin", TEST_SIZE);

	Matrix	testing_batch_output(TEST_SIZE, POSSIBILE_OUTPUTS);

	for (int i = 0; i < TEST_SIZE; i++) {
		for (int j = 0; j < POSSIBILE_OUTPUTS; j++) {
			testing_batch_output.m[i][static_cast<int>(mnist_test_labels[i])] = 1.0;
		}
	}

	Matrix	t_inputs = testing_batch_input.normalize(testing_batch_input.min(), testing_batch_input.max());
	Matrix	t_outputs = testing_batch_output.normalize(testing_batch_output.min(), testing_batch_output.max());

	network.saveConfigJson("network.json");

	Matrix result = network.test(t_inputs);

	int	choices[POSSIBILE_OUTPUTS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	std::cout << "result: " << result << std::endl;
	std::cout << "expected: " << t_outputs << std::endl;
	for (int k = 0; k < TEST_SIZE; k++) {
		double d = 0.0;
		for (int i = 0; i < POSSIBILE_OUTPUTS; i++) {
			d += result.m[k][i];
			std::cout << choices[i] << std::fixed << std::setprecision(2);
			std::cout << " : " << result.m[k][i] * 100 << "% certainty";
			if (t_outputs.m[k][i] == 1)
				std::cout << " (correct)";
			std::cout << std::endl;
		}
		std::cout << "d: " << d << std::endl;
		std::cout << "	---  " << std::endl;
	}

    return (0);
}