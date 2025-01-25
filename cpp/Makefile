CLANG = g++
FLAGS = -Wall -Wextra -Werror -Wshadow -std=c++11
DEBUG_FLAGS = -g -fsanitize=address
OPTIMIZATION_FLAGS = -O3
HEADERS = Matrix.hpp NetworkLayer.hpp NeuralNetwork.hpp HiddenLayer.hpp OutputLayer.hpp math.hpp config.hpp
FILES = main.cpp Matrix.cpp NetworkLayer.cpp NeuralNetwork.cpp HiddenLayer.cpp OutputLayer.cpp math.cpp
OBJ_DIR = obj/
OBJ_FILES = $(addprefix $(OBJ_DIR), $(FILES:.cpp=.opp))
NAME = NeuralNetwork

all: FLAGS += $(OPTIMIZATION_FLAGS)
all: $(OBJ_DIR) $(NAME)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(NAME): $(OBJ_FILES) $(HEADERS)
	$(CLANG) $(FLAGS) $(OBJ_FILES) -o $(NAME)

$(OBJ_DIR)%.opp: %.cpp $(HEADERS)
	$(CLANG) $(FLAGS) -c $< -o $@

debug: FLAGS += $(DEBUG_FLAGS)
debug: $(OBJ_DIR) $(NAME)

clean:
	rm -f $(OBJ_FILES)
	rm -r $(OBJ_DIR)

fclean: clean
	rm -f $(NAME)

re: fclean all

.PHONY: all debug clean fclean re
