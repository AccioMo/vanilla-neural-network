
#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork( int size, int *nodes, double learning_rate )
	: _size(size),
	_learning_rate(learning_rate),
	output_layer(OutputLayer(nodes[size - 2], nodes[size - 1])) {
	this->hidden_layers.reserve(size - 2);
	for (int i = 0; i < size - 2; i++) {
		this->hidden_layers.emplace_back(HiddenLayer(i, nodes[i], nodes[i + 1]));
	}
}

NeuralNetwork::NeuralNetwork( const NeuralNetwork &og )
	: _size(og._size),
	_learning_rate(og._learning_rate),
	output_layer(og.output_layer),
	hidden_layers(og.hidden_layers)
{ }

NeuralNetwork	&NeuralNetwork::operator=( const NeuralNetwork &og ) {
	this->_size = og._size;
	this->_learning_rate = og._learning_rate;
	this->output_layer = og.output_layer;
	this->hidden_layers = og.hidden_layers;
	return (*this);
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

void	NeuralNetwork::update( const Matrix &inputs ) {
	Matrix	outputs = inputs;
	for (auto &layer : hidden_layers) {
		layer.update(outputs, this->_learning_rate);
		outputs = layer.getOutputs();
	}
	output_layer.update(outputs, this->_learning_rate);
}

void	NeuralNetwork::training( Matrix input_batch, Matrix output_batch, int epochs ) {
	for (int age = 0; age < epochs; age++) {
		this->feedforward(input_batch);
		this->backpropagation(output_batch);
		this->update(input_batch);
		if (this->output_layer.getErrors().abs().mean() == 0.0)
			break ;
	}
}

void	NeuralNetwork::saveConfig(const char *filename) const {
    std::ofstream file(filename, std::ios::out);
    
    if (file.is_open()) {
		file << "{\n";
   		file << "\"size\": " << this->_size << "," << std::endl;
		file << "\"learning_rate\": " << this->_learning_rate << ", " << std::endl;
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

Matrix	NeuralNetwork::test( const Matrix input ) {
	this->feedforward(input);
	return (this->output_layer.getOutputs());
}
