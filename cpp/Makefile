CLANG = g++
FLAGS = -Wall -Wextra -Werror -Wshadow -std=c++11
DEBUG_FLAGS = -g -fsanitize=address
OPTIMIZATION_FLAGS = -O3
HEADERS = Matrix.hpp NetworkLayer.hpp NeuralNetwork.hpp HiddenLayer.hpp OutputLayer.hpp math.hpp utils.hpp config.hpp
FILES = main.cpp Matrix.cpp NetworkLayer.cpp NeuralNetwork.cpp HiddenLayer.cpp OutputLayer.cpp math.cpp utils.cpp
OBJ_DIR = obj/
DEBUG_OBJ_DIR = debug_obj/
OBJ_FILES = $(addprefix $(OBJ_DIR), $(FILES:.cpp=.opp))
DEBUG_OBJ_FILES = $(addprefix $(DEBUG_OBJ_DIR), $(FILES:.cpp=_debug.opp))
NAME = nn
NAME_DEBUG = nn_debug

all: FLAGS += $(OPTIMIZATION_FLAGS)
all: $(OBJ_DIR) $(NAME)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(DEBUG_OBJ_DIR):
	@mkdir -p $(DEBUG_OBJ_DIR)

$(NAME): $(OBJ_FILES) $(HEADERS)
	$(CLANG) $(FLAGS) $(OBJ_FILES) -o $(NAME)

$(NAME_DEBUG): $(DEBUG_OBJ_FILES) $(HEADERS)
	$(CLANG) $(FLAGS) $(DEBUG_OBJ_FILES) -o $(NAME_DEBUG)

$(OBJ_DIR)%.opp: %.cpp $(HEADERS)
	$(CLANG) $(FLAGS) -c $< -o $@

$(DEBUG_OBJ_DIR)%_debug.opp: %.cpp $(HEADERS)
	$(CLANG) $(FLAGS) -c $< -o $@

debug: FLAGS += $(DEBUG_FLAGS)
debug: $(DEBUG_OBJ_DIR) $(NAME_DEBUG)

clean:
	rm -f $(OBJ_FILES)
	rm -f $(DEBUG_OBJ_FILES)
	rm -r $(OBJ_DIR)

fclean: clean
	rm -f $(NAME) $(NAME_DEBUG)

re: fclean all

.PHONY: all debug clean fclean re
