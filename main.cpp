
#include "NeuralNetwork.hpp"
#include "NetworkLayer.hpp"
#include "Matrix.hpp"
#include <vector>

int main( int ac, char **av )
{
	std::string config = "configs/20i-92.14%.bin";


	if (ac > 1)
	{
		if (std::string(av[1]) == "train") {
			if (ac > 2) {
				NeuralNetwork	network(av[2]);
				network.trainOnFile("mnist/mnist_train_images.bin", 
									"mnist/mnist_train_labels.bin", 
									config.c_str());
			} else {
				NeuralNetwork	network = NeuralNetwork(3, 
											(int []){784, 128, 10}, 
											0.01, 
											0.001, 
											0.9, 
											0.999);
				network.trainOnFile("mnist/mnist_train_images.bin", 
									"mnist/mnist_train_labels.bin", 
									config.c_str());
			}
		} else if (std::string(av[1]) == "test") {
			NeuralNetwork	network = NeuralNetwork(config.c_str());
			network.testOnFile("mnist/mnist_test_images.bin", "mnist/mnist_test_labels.bin");
		} else {
			if (ac > 2) {
				NeuralNetwork	network(av[2]);
				network.runOnImage(av[1]);
			} else {
				NeuralNetwork	network = NeuralNetwork(config.c_str());
				network.runOnImage(av[1]);
			}
		}
	}
	else {
		std::cerr << "usage: ./nn <image_path>" << std::endl;
	}

    return (0);
}
