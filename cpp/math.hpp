
#ifndef MATH_HPP
# define MATH_HPP

# include "Matrix.hpp"
# include <cmath>

Matrix	ReLU( const Matrix &input );
Matrix	sigmoid( const Matrix &input );
Matrix	sigmoid_derivative( const Matrix &input );
double	xavier_glorot_init(int fan_in, int fan_out);

#endif
