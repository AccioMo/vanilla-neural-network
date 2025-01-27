
#ifndef UTILS_HPP
# define UTILS_HPP

# include <sys/time.h>
# include <fstream>

# include "config.hpp"
# include "Matrix.hpp"

double	ft_get_time(void);
std::vector<Matrix>	get_input_batch( const char *filename );
std::vector<Matrix>	get_input_labels( const char *filename );

#endif
