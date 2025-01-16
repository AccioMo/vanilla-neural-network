
#ifndef MATRIX_HPP
# define MATRIX_HPP

# include <iostream>
#include <vector>

class Matrix {
	private:
		int	_rows;
		int	_columns;

	public:
		std::vector<std::vector<double>>	m;

		Matrix( std::vector<std::vector<double>> array );
		Matrix( int rows, int columns );
		Matrix	&operator=( const Matrix &og );
		Matrix( const Matrix &og );
		~Matrix();

		int	rows( void ) const;
		int	columns( void ) const;

		Matrix	operator+( const Matrix &m2 ) const;
		Matrix	operator*( const Matrix &og ) const;

		Matrix	transpose( void ) const;
};

std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

#endif
