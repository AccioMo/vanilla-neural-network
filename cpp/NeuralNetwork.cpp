
#include "NeuralNetwork.hpp"

# define STB_IMAGE_IMPLEMENTATION
# include "stb_image.h"

NeuralNetwork::NeuralNetwork( void ) : NeuralNetwork(0, (int []){})
 { }

NeuralNetwork::NeuralNetwork( int size,
							int *nodes,
							double learning_rate,
							double l2_lambda,
							double beta1,
							double beta2 )
	: _size(size),
	_learning_rate(learning_rate),
	_l2_lambda(l2_lambda),
	_beta1(beta1),
	_beta2(beta2) {
	if (size < 2)
		return ;
	this->output_layer = OutputLayer(nodes[size - 2], nodes[size - 1]);
	this->hidden_layers.reserve(size - 2);
	for (int i = 0; i < size - 2; i++) {
		this->hidden_layers.emplace_back(HiddenLayer(i, nodes[i], nodes[i + 1]));
	}
}

NeuralNetwork::NeuralNetwork( const char *filename ) : NeuralNetwork(0, (int []){}) {

	std::streamsize size = get_file_size(filename);
	std::vector<unsigned char> mnist_train_images = read_binary_file(filename, (size_t)size);
	
	int	i = 0;
	
	std::memcpy(&this->_size, &mnist_train_images[i], sizeof(int));
	i += sizeof(int);
	std::memcpy(&this->_learning_rate, &mnist_train_images[i], sizeof(double));
	i += sizeof(double);

	int	*config_nodes = new int[this->_size];
	for (int j = 0; j < this->_size; j++) {
		std::memcpy(&config_nodes[j], &mnist_train_images[i], sizeof(int));
		i += sizeof(int);
	}

	for (int k = 0; k < this->_size - 2; k++) {

		int		neurons = config_nodes[k + 1];

		Matrix	weights = Matrix(config_nodes[k], neurons);
		for (int j = 0; j < config_nodes[k]; j++) {
			for (int l = 0; l < neurons; l++) {
				std::memcpy(&weights.m[j][l], &mnist_train_images[i], sizeof(double));
				i += sizeof(double);
			}
		}
		Matrix	biases = Matrix(1, neurons);

		for (int j = 0; j < neurons; j++) {
			std::memcpy(&biases.m[0][j], &mnist_train_images[i], sizeof(double));
			i += sizeof(double);
		}

		HiddenLayer	hidden_layer(k, config_nodes[k], neurons);
		hidden_layer.setWeights(weights);
		hidden_layer.setBiases(biases);
		hidden_layer.setSize(neurons);
		this->hidden_layers.push_back(hidden_layer);
	}

	int		neurons = config_nodes[this->_size - 1];

	Matrix	weights = Matrix(config_nodes[this->_size - 2], neurons);

	for (int j = 0; j < config_nodes[this->_size - 2]; j++) {
		for (int l = 0; l < neurons; l++) {
			std::memcpy(&weights.m[j][l], &mnist_train_images[i], sizeof(double));
			i += sizeof(double);
		}
	}

	Matrix	biases = Matrix(1, neurons);
	
	for (int j = 0; j < neurons; j++) {
		std::memcpy(&biases.m[0][j], &mnist_train_images[i], sizeof(double));
		i += sizeof(double);
	}

	OutputLayer	new_output_layer(config_nodes[this->_size - 2], neurons);
	new_output_layer.setWeights(weights);
	new_output_layer.setBiases(biases);
	new_output_layer.setSize(neurons);
	this->output_layer = new_output_layer;
}

NeuralNetwork::~NeuralNetwork() { }

void	NeuralNetwork::feedforward( const Matrix &inputs ) {
	Matrix	outputs = inputs;
	for (auto &layer : hidden_layers) {
		outputs = layer.feedforward(outputs);
	}
	output_layer.feedforward(outputs);
}

void	NeuralNetwork::backpropagation( const Matrix &expected_outputs ) {
	NetworkLayer	*next_layer = &output_layer;
	output_layer.backpropagation(expected_outputs);
	for (int i = hidden_layers.size() - 1; i >= 0; i--) {
		this->hidden_layers[i].backpropagation(*next_layer);
		next_layer = &this->hidden_layers[i];
	}
}

void	NeuralNetwork::update( const Matrix &inputs, int timestep ) {
	Matrix	outputs = inputs;
	for (auto &layer : hidden_layers) {
		layer.update(outputs, this->_learning_rate, timestep, \
			this->_l2_lambda, this->_beta1, this->_beta2);
		outputs = layer.getOutputs();
	}
	output_layer.update(outputs, this->_learning_rate, timestep, \
		this->_l2_lambda, this->_beta1, this->_beta2);
}

void	NeuralNetwork::train( Matrix input_batch, Matrix output_batch, int epochs, int timestep ) {
	for (int age = 0; age < epochs; age++) {
		this->feedforward(input_batch);
		this->backpropagation(output_batch);
		this->update(input_batch, timestep);
		if (age == 0)
			std::cout << "accuracy (start): " << std::fixed << std::setprecision(2) \
				<< this->calculateAccuracy(output_batch).mean() * 100.0 << "%" << std::endl;
		std::cout << "\033[2Kepochs\t\t: " << age << std::endl << "\033[A\r";
	}
}

void	NeuralNetwork::trainOnFile( const char *filename, const char *labels, const char *output_file ) {

	std::vector<Matrix>	inputs = get_input_batch(filename);
	std::vector<Matrix>	outputs = get_input_labels(labels);

	std::cout << "Network constructed!" << std::endl;

	double start = ft_get_time();
	std::cout << std::endl << "   --- TRAINING ---	" << std::endl;
	int	total_iterations = TRAIN_SIZE / BATCH_SIZE;
	for (int i = 0; i < total_iterations; i++) {
		std::cout << "Iteration " << i + 1 << " of " << total_iterations << std::endl;
		Matrix	normalized_inputs = inputs[i].normalize(INPUT_MIN, INPUT_MAX);
		Matrix	normalized_outputs = outputs[i].normalize(OUTPUT_MIN, OUTPUT_MAX);
		this->train(normalized_inputs, normalized_outputs, EPOCHS, i + 1);
		this->printData(normalized_outputs);
		std::cout << "   ---	" << std::endl;
	}

	std::cout << "Training done!" << std::endl << std::endl;
	std::cout << "time		: " << (ft_get_time() - start) / 1000 << "s" << std::endl;
	std::cout << "   ---	" << std::endl;

	this->saveConfigBin(output_file);

	std::cout << "Network saved!" << std::endl;
}

void	NeuralNetwork::test( const Matrix input, const Matrix expected_outputs ) {
	this->feedforward(input);
	this->printData(expected_outputs);
}

void	NeuralNetwork::testOnFile( const char *filename, const char *labels ) {

	std::vector<Matrix>	t_inputs = get_input_batch(filename);
	std::vector<Matrix>	t_outputs = get_input_labels(labels);

	std::cout << "   --- TESTING ---" << std::endl;

	double	accuracy = 0.0;
	int	test_iterations = TEST_SIZE / BATCH_SIZE;
	for (int i = 0; i < test_iterations; i++) {
		std::cout << "Iteration " << i + 1 << " of " << test_iterations << std::endl;
		Matrix	normalized_inputs = t_inputs[i].normalize(INPUT_MIN, INPUT_MAX);
		Matrix	normalized_outputs = t_outputs[i].normalize(OUTPUT_MIN, OUTPUT_MAX);
		this->feedforward(normalized_inputs);
		this->backpropagation(normalized_outputs);
		this->printData(normalized_outputs);
		accuracy += this->calculateAccuracy(normalized_outputs).mean();
		std::cout << "   ---	" << std::endl;
	}
	accuracy /= test_iterations + 1;
	std::cout << "TEST ACCURACY\t: " << accuracy * 100 << "%" << std::endl;
}

Matrix	NeuralNetwork::run( const Matrix input ) {
	Matrix	normalized_input = input.normalize(INPUT_MIN, INPUT_MAX);

	this->feedforward(normalized_input);

	int	possible_outputs[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	for (int i = 0; i < POSSIBILE_OUTPUTS; i++) {
		std::cout << possible_outputs[i] << ": " << std::fixed << std::setprecision(2) << this->output_layer.getOutputs().m[0][i] * 100 << "%" << std::endl;
	}
	return (this->output_layer.getOutputs());
}

Matrix	NeuralNetwork::runOnImage( const char *filename ) {
	int	width, height, channels;

	unsigned char	*image = stbi_load(filename, &width, &height, &channels, 1);
	if (image == NULL) {
		std::cerr << "Error loading image" << std::endl;
		return (Matrix());
	}
	if (width != 28 || height != 28) {
		std::cerr << "Image must be 28x28" << std::endl;
		return (Matrix());
	}

	Matrix	input(1, IMAGE_SIZE);
	for (int j = 0; j < IMAGE_SIZE; j++) {
		if (image[j] > 255 * 0.85) 
			input.m[0][j] = 0.0;
		else
			input.m[0][j] = static_cast<double>(255 - image[j]);
	}
	
	Matrix	output = this->run(input);

	stbi_image_free(image);
	return (output);
}

void	NeuralNetwork::saveConfigJson( const char *filename ) const {
    std::ofstream file(filename, std::ios::out);
    
    if (file.is_open()) {
		file << "{\n";
   		file << "\"size\": " << this->_size << "," << std::endl;
		file << "\"learning_rate\": " << this->_learning_rate << ", " << std::endl;
		file << "\"layers\": [" << this->hidden_layers[0].getWeights().rows() << ", ";
		for (auto &layer_neurons : this->hidden_layers)
			file << layer_neurons.getSize() << ", ";
		file << this->output_layer.getSize();
		file << "], " << std::endl;

		file << "\"hidden_layers\": [\n";
		for (auto &layer : this->hidden_layers) {
			file << "{" << std::endl;
			file << "\"weights\": " << layer.getWeights() << ", " << std::endl;
			file << "\"biases\": " << layer.getBiases() << std::endl;
			file << "}";
			if (&layer != &this->hidden_layers.back())
				file << std::endl << "," << std::endl;
		}
		file << "],\n";
		file << "\"output_layer\": {" << std::endl;
		file << "\"weights\": " << this->output_layer.getWeights() << ", " << std::endl;
		file << "\"biases\": " << this->output_layer.getBiases() << std::endl;
		file << "}";
		file << "}" << std::endl;
        file.close();
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
}

void	NeuralNetwork::saveConfigBin(const char *filename) const {
	std::ofstream file(filename, std::ios::binary);

    if (file.is_open()) {

        file.write(reinterpret_cast<const char *>(&this->_size), sizeof(this->_size));

        file.write(reinterpret_cast<const char *>(&this->_learning_rate), sizeof(this->_learning_rate));

        int rows = this->hidden_layers[0].getWeights().rows();
        file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));

        for (const auto &layer : this->hidden_layers) {
            int size = layer.getSize();
            file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        }

        int output_size = this->output_layer.getSize();
        file.write(reinterpret_cast<const char *>(&output_size), sizeof(output_size));

        for (const auto &layer : this->hidden_layers) {
			for (int i = 0; i < layer.getWeights().rows(); i++) {
				file.write(reinterpret_cast<const char *>(layer.getWeights().m[i].data()), layer.getWeights().m[i].size() * sizeof(double));
			}
			file.write(reinterpret_cast<const char *>(layer.getBiases().m[0].data()), layer.getBiases().m[0].size() * sizeof(double));
        }

        for (int i = 0; i < this->output_layer.getWeights().rows(); i++) {
			file.write(reinterpret_cast<const char *>(this->output_layer.getWeights().m[i].data()), this->output_layer.getWeights().m[i].size() * sizeof(double));
		}
		file.write(reinterpret_cast<const char *>(this->output_layer.getBiases().m[0].data()), this->output_layer.getBiases().m[0].size() * sizeof(double));

        file.close();
    } else {
        std::cerr << "Error opening file: " << filename;
    }
}

void	NeuralNetwork::printData( const Matrix expected_outputs ) const {
	double max_entropy = -std::log(1.0 / (double)POSSIBILE_OUTPUTS);
	std::cout << "accuracy (end)\t: " << this->calculateAccuracy(expected_outputs).mean() * 100 << "%" << std::endl;
	std::cout << "entropy\t\t: " << this->calculateEntropy().mean() << " (max " << max_entropy << ")" << std::endl;
	std::cout << "confidence\t: " << (1.0 - (this->calculateEntropy().mean() / max_entropy)) * 100 << "%" << std::endl;
}

Matrix	NeuralNetwork::calculateEntropy( void ) const {
	/* entropy = -sum(p * log(p)) */
	double	epsilon = 1e-15;
	Matrix	predicted_outputs = this->output_layer.getOutputs();
	Matrix	entropy = predicted_outputs.hadamard_product(log(predicted_outputs + epsilon)).sum_rows() * -1.0;
	return (entropy);
}

Matrix	NeuralNetwork::calculateAccuracy( const Matrix &expected_ouputs ) const {
	/* assuming softmax activation */
	Matrix	predicted_ouputs = this->output_layer.getOutputs().argmax();
	return (predicted_ouputs == expected_ouputs.argmax());
}

Matrix	NeuralNetwork::getEntropy( void ) const {
	return (_entropy);
}

Matrix	NeuralNetwork::getConfidence( void ) const {
	return (_confidence);
}

double	NeuralNetwork::getLearningRate( void ) const {
	return (_learning_rate);
}

void	NeuralNetwork::setEntropy( Matrix entropy  ) {
	this->_entropy = entropy;
}

void	NeuralNetwork::setConfidence( Matrix confidence  ) {
	this->_confidence = confidence;
}

void	NeuralNetwork::setLearningRate( double learning_rate ) {
	this->_learning_rate = learning_rate;
}
