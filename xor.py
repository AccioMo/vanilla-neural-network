import numpy as np

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

weights_1 = np.random.randn(2, 4) * 0.1
weights_2 = np.random.randn(4, 1) * 0.1

bias_1 = np.zeros((1, 4))
bias_2 = np.zeros((1, 1))

# weights_1 = np.array([[4.00299883, 7.39770269, 1.39731008, 2.95323141],
#  [4.00107995, 7.40073565, 1.39936918, 2.95404757]])
# weights_2 = np.array([[-8.15147657],
#  [13.28615313],
#  [-3.8908669 ],
#  [-5.85208247]])
# bias_1 = np.array([[-6.30147609, -3.48884086, -2.43858306, -4.77348638]])
# bias_2 = np.array([[-5.09586507]])

learning_rate = 0.5

error_margin = 0.0

def calculate_success_rate(error):
    thresholded_outputs = (error >= 0.5).astype(int)
    correct_predictions = (thresholded_outputs == outputs).astype(int)
    success_rate = (success_rate + np.mean(correct_predictions)) / 2
    return success_rate

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

while True:
    # Feed Forward (i think?)
    hidden_inputs = np.dot(inputs, weights_1) + bias_1
    hidden_outputs = sigmoid(hidden_inputs)

    output_inputs = np.dot(hidden_outputs, weights_2) + bias_2
    output_outputs = sigmoid(output_inputs)

    # Backpropagation (probably)
    error = outputs - output_outputs # loss_function(output_outputs)
    output_delta = error * sigmoid_derivative(output_outputs)

    hidden_error = np.dot(output_delta, weights_2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_outputs)

    error_magnitude = 1 - np.mean(np.abs(error))

    print(f"Error Magnitude: {(error_magnitude * 100):.2f}%", end='\r')
    if error_magnitude >= 0.995:
        break

    # Training
    weights_1 += np.dot(inputs.T, hidden_delta) * learning_rate
    weights_2 += np.dot(hidden_outputs.T, output_delta) * learning_rate

    bias_1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
    bias_2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

new_inputs = [0, 1]

print("\nWeights 1:", weights_1)
print("Weights 2:", weights_2)
print("Bias 1:", bias_1)
print("Bias 2:", bias_2)

hidden_inputs = np.dot(new_inputs, weights_1) + bias_1
hidden_outputs = sigmoid(hidden_inputs)

output_inputs = np.dot(hidden_outputs, weights_2) + bias_2
output_outputs = sigmoid(output_inputs)

print(output_outputs)
