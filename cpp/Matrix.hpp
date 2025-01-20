
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
		Matrix( int rows, int columns, double rand_range );
		Matrix	&operator=( const Matrix &og );
		Matrix( const Matrix &og );
		~Matrix();

		int	rows( void ) const;
		int	columns( void ) const;

		/* default matrix addition and subtraction. if the matrices 
		are of different sizes it attempts a broadcast */
		Matrix	operator+( const Matrix &to_add ) const;
		Matrix	operator-( const Matrix &to_subtract ) const;

		/* default matrix multiplication. if the matrices 
		are not aligned it applies hadamard product */
		Matrix	operator*( const Matrix &mult ) const;

		/* default matrix scalar multiplication */
		Matrix	operator*( const double mult ) const;

		/* sums the matrix along its columns (collapsing rows) */
		Matrix	sum_columns( void ) const;

		/* matrix transpose */
		Matrix	transpose( void ) const;

		/* repeats columns/rows `n` times. used in broadcast */
		Matrix	repeat_columns( int n ) const;
		Matrix	repeat_rows( int n ) const;

		/* hadamard product. used in multiplication overload */
		Matrix	hadamard_product( const Matrix &mult ) const;

		/* calculates the arithmetic mean of the matrix */
		double	mean( void ) const;

		/* returns the absolute value of a matrix */
		Matrix	abs( void ) const;

		/* normalizes the matrix using min-max scaling */
		Matrix	normalize( double min, double max ) const;

		/* reverse normalizes the matrix using min-max scaling */
		Matrix	denormalize( double min, double max ) const;

		/* returns minimum value in the matrix */
		double	min() const;

		/* returns maximum value in the matrix */
		double	max() const;
};

std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

#endif
