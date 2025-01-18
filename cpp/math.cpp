
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
