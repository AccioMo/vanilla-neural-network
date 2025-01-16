
// #include "NeuralNetwork.hpp"
// #include "NetworkLayer.hpp"
#include "Matrix.hpp"
#include <vector>

int main( void )
{
	std::vector<std::vector<double>>	a{{1, 2, 3, 5},
										  {4, 5, 6, 7},
										  {7, 8, 9, 10}};

	std::vector<std::vector<double>>	b{{1, 2, 3},
										  {4, 5, 6},
										  {7, 8, 9},
										  {10, 11, 12}};

	Matrix w1(a);
	Matrix w2(b);

	std::cout << "rows: " << w1.rows() << "; columns: " << w1.columns() << std::endl;

	Matrix c = w1 * w2;

	std::cout << c << std::endl;

    return (0);
}