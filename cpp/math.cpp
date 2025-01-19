
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
			element = 1.0 / (1.0 + std::exp(-element));
		}
	}
	return (output);
}

Matrix	sigmoid_derivative( const Matrix &input ) {
	Matrix	output = input;
	for (auto &row : output.m) {
		for (auto &element : row) {
			element = element * (1.0 - element);
		}
	}
	return (output);
}

double	xavier_glorot_init(int fan_in, int fan_out) {
	return (4.0 * sqrt(6.0 / (double)(fan_in + fan_out)));
}
