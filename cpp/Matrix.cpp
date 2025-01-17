
#include "Matrix.hpp"

Matrix::Matrix( std::vector<std::vector<double>> array ) : m(array) {
	this->_rows = array.size();
	if (this->_rows == 0)
			throw std::invalid_argument("empty array");
	this->_columns = array[0].size();
	for (int i = 0; i < this->_rows; i++) {
		if (this->_columns != (int)array[i].size())
			throw std::invalid_argument("inconsistent size");
	}
}

Matrix::Matrix( int rows, int columns ) {
	this->_rows = rows;
	this->_columns = columns;
	this->m = std::vector<std::vector<double>>(rows, std::vector<double>(columns));
}

Matrix::Matrix( int rows, int columns, t_starting_value starting_value ) {
	this->_rows = rows;
	this->_columns = columns;
	this->m = std::vector<std::vector<double>>(rows, std::vector<double>(columns));
	if (starting_value == M_RAND) {
		std::random_device					rd;
		std::mt19937						gen(rd());
		std::uniform_real_distribution<>	dis(-3.0, 3.0);
		for ( auto &row : this->m ) {
			for (auto &value : row) {
				value = dis(gen);
			}
		}
	}
}

Matrix::Matrix( const Matrix &og ) {
	this->_rows = og._rows;
	this->_columns = og._columns;
	this->m = og.m;
}

Matrix::~Matrix() { }


int	Matrix::rows( void ) const {
	return this->_rows;
}

int	Matrix::columns( void ) const {
	return this->_columns;
}


Matrix	&Matrix::operator=( const Matrix &og ) {
	this->_rows = og._rows;
	this->_columns = og._columns;
	this->m = og.m;
	return *this;
}

Matrix	Matrix::operator+( const Matrix &m1 ) const {
	if (m1.rows() != this->rows() || m1.columns() != this->columns())
		throw std::invalid_argument("different sizes");
	Matrix result(m1);
	for (int i = 0; i < m1.rows(); i++) {
		for (int j = 0; j < m1.columns(); j++) {
			result.m[i][j] += m1.m[i][j];
		}
	}
	return (result);
}

Matrix	Matrix::operator-( const Matrix &m1 ) const {
	if (m1.rows() != this->rows() || m1.columns() != this->columns())
		throw std::invalid_argument("different sizes");
	Matrix result(m1);
	for (int i = 0; i < m1.rows(); i++) {
		for (int j = 0; j < m1.columns(); j++) {
			result.m[i][j] -= m1.m[i][j];
		}
	}
	return (result);
}

Matrix	Matrix::operator*( const Matrix &mult ) const {
	if (mult.rows() != this->columns() || mult.columns() != this->rows())
		throw std::invalid_argument("matrices not aligned");
	Matrix result(this->rows(), mult.columns());
	for (int i = 0; i < this->rows(); i++) {
		for (int j = 0; j < this->rows(); j++) {
			for (int k = 0; k < this->columns(); k++) {
				result.m[i][j] += this->m[i][k] * mult.m[k][j];
			}
		}
	}
	return (result);
}

Matrix	Matrix::transpose( void ) const {
	Matrix	T(this->columns(), this->rows());
	for (int i = 0; i < this->rows(); i++) {
		for (int j = 0; j < this->columns(); j++) {
			T.m[j][i] = this->m[i][j];
		}
	}
	return (T);
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
	os << "[";
    for (int i = 0; i < matrix.rows(); ++i) {
		if (i != 0) os << " ";
		os << "[";
        for (int j = 0; j < matrix.columns(); ++j) {
            os << matrix.m[i][j];
			if (j != matrix.columns() - 1) os << " ";
        }
		os << "]";
        if (i != matrix.rows() - 1) os << "\n";
    }
	os << "]";
    return os;
}
