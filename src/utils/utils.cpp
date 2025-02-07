
#include "utils.hpp"

double	ft_get_time(void) {
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
        std::cout << "Error opening file: " << filename << std::endl;
    }
	return (outputs);
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

std::streamsize get_file_size(const char *filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        std::streamsize size = file.tellg();
        file.close();
        return size;
    } else {
        throw std::runtime_error("Unable to open file");
    }
}
