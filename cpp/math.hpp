
#ifndef MATH_HPP
# define MATH_HPP

# include "Matrix.hpp"
# include <cmath>

Matrix	ReLU( const Matrix &input );
Matrix	sigmoid( const Matrix &input );
Matrix	sigmoid_derivative( const Matrix &input );

#endif
