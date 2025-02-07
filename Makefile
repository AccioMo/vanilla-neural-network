CLANG = g++
FLAGS = -Wall -Wextra -Werror -Wshadow -std=c++11
DEBUG_FLAGS = -g -fsanitize=address
OPTIMIZATION_FLAGS = -flto -O3
HEADERS = Matrix.hpp NetworkLayer.hpp NeuralNetwork.hpp HiddenLayer.hpp OutputLayer.hpp math.hpp utils.hpp config.hpp
BASE_FILES = NetworkLayer.cpp NeuralNetwork.cpp HiddenLayer.cpp OutputLayer.cpp
MATH_FILES = math.cpp Matrix.cpp 
UTILS_FILES = utils.cpp
MAIN_FILE = main.cpp

INCLUDE_DIR = include/
SRC_DIR = src/
BASE_DIR = $(SRC_DIR)base/
MATH_DIR = $(SRC_DIR)math/
UTILS_DIR = $(SRC_DIR)utils/
OBJ_DIR = obj/
DEBUG_OBJ_DIR = debug_obj/

INCLUDES = $(addprefix $(INCLUDE_DIR), $(HEADERS))
FILES = $(BASE_FILES) $(MATH_FILES) $(UTILS_FILES) $(MAIN_FILE)
OBJ_FILES = $(addprefix $(OBJ_DIR), $(FILES:.cpp=.opp))
DEBUG_OBJ_FILES = $(addprefix $(DEBUG_OBJ_DIR), $(FILES:.cpp=_debug.opp))
NAME_DEBUG = nn_debug
NAME = nn

all: FLAGS += $(OPTIMIZATION_FLAGS)
all: $(OBJ_DIR) $(NAME)

# ==== RELEASE ==== #
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(NAME): $(OBJ_FILES) $(INCLUDES)
	$(CLANG) $(FLAGS) $(OBJ_FILES) -o $(NAME)

$(OBJ_DIR)%.opp: $(BASE_DIR)%.cpp $(INCLUDES)
	$(CLANG) $(FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(OBJ_DIR)%.opp: $(MATH_DIR)%.cpp $(INCLUDES)
	$(CLANG) $(FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(OBJ_DIR)%.opp: $(UTILS_DIR)%.cpp $(INCLUDES)
	$(CLANG) $(FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(OBJ_DIR)%.opp: %.cpp $(INCLUDES)
	$(CLANG) $(FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# ==== DEBUG ==== #
$(DEBUG_OBJ_DIR):
	@mkdir -p $(DEBUG_OBJ_DIR)

$(NAME_DEBUG): $(DEBUG_OBJ_FILES) $(HEADERS)
	$(CLANG) $(FLAGS) $(DEBUG_OBJ_FILES) -o $(NAME_DEBUG)

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
