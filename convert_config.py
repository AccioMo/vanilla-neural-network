import json
import struct

def json_to_binary(json_filename, binary_filename):
	with open(json_filename, 'r') as json_file:
		data = json.load(json_file)

	with open(binary_filename, 'wb') as binary_file:
		# Write size
		binary_file.write(struct.pack('i', data['size']))

		# Write learning rate
		binary_file.write(struct.pack('d', data['learning_rate']))

		for layer_neurons in data['layers']:
			binary_file.write(struct.pack('i', layer_neurons))

		# Write hidden layers
		for layer in data['hidden_layers']:

			# Write weights
			weights = layer['weights']
			for weight_row in weights:
				for weight in weight_row:
					binary_file.write(struct.pack('d', weight))

			# Write biases
			biases = layer['biases']
			for bias_row in biases:
				for bias in bias_row:
			  		binary_file.write(struct.pack('d', bias))

		# Write output layer
		output_layer = data['output_layer']

		# Write weights
		weights = output_layer['weights']
		for weight_row in weights:
			for weight in weight_row:
				binary_file.write(struct.pack('d', weight))

		# Write biases
		biases = output_layer['biases']
		for bias_row in biases:
			for bias in bias_row:
				binary_file.write(struct.pack('d', bias))

# Example usage
json_to_binary('cpp/60k-005-40i.json', 'cpp/60k-005-40i.bin')