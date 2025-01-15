# import numpy
import random
import os
#initialize learning rate, bias, and weights
learning_rate = 1 
bias = 1
# Random weights
weights = [random.random(),random.random(),random.random()]

def loss_function(expected_output, actual_output):
   loss = expected_output - actual_output
   return loss

def activation_function(expected_output, actual_output): # Heaviside step
    if actual_output > 0:
      actual_output = 1
    else:
      actual_output = 0
    return loss_function(expected_output, actual_output)

def perceptron(input1, input2, expected_output):
   actual_output = input1 * weights[0] + input2 * weights[1] + bias * weights[2]

   error = activation_function(expected_output, actual_output)
   
   #update weights
   weights[0] += error * input1 * learning_rate
   weights[1] += error * input2 * learning_rate
   weights[2] += error * bias * learning_rate
   return error == 0

print(weights)
# training
loss = 0.5
for i in range(5) :
   loss = (loss + perceptron(1,1,1)) / 2
   loss = (loss + perceptron(1,0,1)) / 2
   loss = (loss + perceptron(0,1,1)) / 2
   loss = (loss + perceptron(0,0,0)) / 2
   print(f"{loss * 100:.2f}%")
   
print(weights)
x = int(input("enter a value for x: "))
y = int(input("enter a value for y: "))
output_perceptron = x * weights[0] + y * weights[1] + bias * weights[2]
if output_perceptron > 0 :
   output_perceptron = 1
else :
   output_perceptron = 0
print( "Output is : ", output_perceptron)