CLANG = c++
FLAGS = -Wall -Wextra -Werror -Wshadow -std=c++11 -g -fsanitize=address
HEADERS = Matrix.hpp NetworkLayer.hpp NeuralNetwork.hpp HiddenLayer.hpp OutputLayer.hpp math.hpp
FILES = main.cpp Matrix.cpp NetworkLayer.cpp NeuralNetwork.cpp HiddenLayer.cpp OutputLayer.cpp math.cpp
OBJ_DIR = obj/
OBJ_FILES = $(addprefix $(OBJ_DIR), $(FILES:.cpp=.opp))
NAME = NeuralNetwork

all: $(OBJ_DIR) $(NAME)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(NAME): $(OBJ_FILES) $(HEADERS)
	$(CLANG) $(FLAGS) $(OBJ_FILES) -o $(NAME)

$(OBJ_DIR)%.opp: %.cpp $(HEADERS)
	$(CLANG) $(FLAGS) -c $< -o $@

clean:
	rm -f $(OBJ_FILES)
	rmdir $(OBJ_DIR) 2>/dev/null || True

fclean: clean
	rm -f $(NAME)

re: fclean all

.PHONY: all clean fclean re
