
#include "NeuralNetwork.hpp"
#include "NetworkLayer.hpp"
#include "Matrix.hpp"
#include <vector>
#include <iomanip>
#include <fstream>

const int NETWORK_SIZE = 3;
const int HIDDEN_LAYERS = NETWORK_SIZE - 2;

const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;
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

	NeuralNetwork	network(NETWORK_SIZE, (int *)nodes, 0.5);

	std::cout << "Network constructed!" << std::endl;

	// std::cout << "--- PRE TRAINING ---" << std::endl;
	// std::cout << network.hidden_layers[0] << std::endl;
	// std::cout << network.output_layer << std::endl;

	double start = ft_get_time();
	network.training(inputs, outputs, 1);

	// std::cout << "--- POST TRAINING ---" << std::endl;
	// std::cout << network.hidden_layers[0] << std::endl;
	// std::cout << network.output_layer << std::endl;
	// std::cout << "errors	: " << network.output_layer.getErrors() << std::endl;
	// std::cout << "outputs	: " << network.output_layer.getOutputs() << std::endl;
	std::cout << "accuracy	: " << (1 - network.output_layer.getErrors().abs().mean()) * 100 << "%" << std::endl;
	std::cout << "time		: " << (ft_get_time() - start) / 1000 << "s" << std::endl;

	// Matrix result = network.test(test_inputs);

	// Matrix	output_options((t_vec){{0.0, 1.0, 2.0}});
	// output_options = output_options.normalize(output_options.min(), output_options.max());

	// for (int i = 0; i < (int)output_options.m[0].size(); i++) {
	// 	result = result.denormalize(raw_outputs.min(), raw_outputs.max());
	// 	network.backpropagation((t_vec){{output_options.m[0][i]}});
	// 	double	loss = network.output_layer.getErrors().abs().mean();
	// 	if (i == 0)
	// 		std::cout << "horizontal with " << std::fixed << std::setprecision(2) << (1 - loss) * 100 << "% certainty" << std::endl;
	// 	else if (i == 1)
	// 		std::cout << "vertical with " << std::fixed << std::setprecision(2) << (1 - loss) * 100 << "% certainty" << std::endl;
	// 	else if (i == 2)
	// 		std::cout << "cross with " << std::fixed << std::setprecision(2) << (1 - loss) * 100 << "% certainty" << std::endl;
	// }

    return (0);
}