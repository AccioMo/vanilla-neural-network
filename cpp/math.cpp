
#include "math.hpp"

Matrix	ReLU( const Matrix &input ) {
	Matrix	output = input;
	for (auto &row : output.m) {
		for (auto &element : row) {
			element = fmax(0, element);
		}
	}
	return (output);
}

Matrix	sigmoid( const Matrix &input ) {
	Matrix	output = input;
	for (auto &row : output.m) {
		for (auto &element : row) {
			element = 1 / (1 + std::exp(-element));
		}
	}
	return (output);
}

Matrix	sigmoid_derivative( const Matrix &input ) {
	Matrix	output = input;
	for (auto &row : output.m) {
		for (auto &element : row) {
			element = 1 * (1 - element);
		}
	}
	return (output);
}
