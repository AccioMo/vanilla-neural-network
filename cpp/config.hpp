
#ifndef CONFIG_HPP
# define CONFIG_HPP

const int NETWORK_SIZE = 3;
const double LEARNING_RATE = 0.001;

const int EPOCHS = 30;
const int IMAGE_SIZE = 28 * 28;
const int POSSIBILE_OUTPUTS = 10;

const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;

/* for SGD (stochastic gradient 
descent) using sequential data 
sampling */
const int BATCH_SIZE = 256;

const double INPUT_MIN = 0.0;
const double INPUT_MAX = 255.0;

const double OUTPUT_MIN = 0.0;
const double OUTPUT_MAX = 1.0;

const double L2_LAMBDA_REGULARIZATION = 0.0001;

#endif
