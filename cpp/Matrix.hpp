
#ifndef MATRIX_HPP
# define MATRIX_HPP

# include <iostream>
# include <vector>
# include <random>

typedef enum e_starting_value {
	M_ZEROS,
	M_RAND
}	t_starting_value;

class Matrix {
	private:
		int	_rows;
		int	_columns;

	public:
		std::vector<std::vector<double>>	m;

		Matrix( void );
		Matrix( std::vector<std::vector<double>> array );
		Matrix( int rows, int columns );
		Matrix( int rows, int columns, t_starting_value starting_value );
		Matrix	&operator=( const Matrix &og );
		Matrix( const Matrix &og );
		~Matrix();

		int	rows( void ) const;
		int	columns( void ) const;

		Matrix	operator+( const Matrix &m2 ) const;
		Matrix	operator-( const Matrix &m2 ) const;
		Matrix	operator*( const Matrix &og ) const;

		Matrix	transpose( void ) const;
		Matrix	repeat_columns( int columns ) const;
		Matrix	repeat_rows( int rows ) const;
};

std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

#endif
