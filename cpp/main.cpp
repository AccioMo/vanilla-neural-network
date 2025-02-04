
#include "NeuralNetwork.hpp"
#include "NetworkLayer.hpp"
#include "Matrix.hpp"
#include <vector>

int main( int ac, char **av )
{
	// NeuralNetwork	network = NeuralNetwork(3, (int []){784, 128, 10});
	NeuralNetwork	network = NeuralNetwork("configs/60k-004-20i.bin");

	if (ac > 1)
		network.runOnImage(av[1]);
	else {
		network.trainOnFile("mnist/mnist_train_images.bin", "mnist/mnist_train_labels.bin", "configs/60k-004-30i-l2.bin");
		// std::cerr << "usage: ./NeuralNetwork <image_path>" << std::endl;
	}

    return (0);
}
