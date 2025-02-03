
#include "NeuralNetwork.hpp"
#include "NetworkLayer.hpp"
#include "Matrix.hpp"
#include <vector>

int main( int ac, char **av )
{
	NeuralNetwork	network = NeuralNetwork("configs/60k-004-80i.bin");

	if (ac > 1)
		network.runOnImage(av[1]);
	else {
		network.trainOnFile("mnist/mnist_train_images.bin", "mnist/mnist_train_labels.bin", "configs/60k-004-120i.bin");
		// std::cerr << "usage: ./NeuralNetwork <image_path>" << std::endl;
	}

    return (0);
}
