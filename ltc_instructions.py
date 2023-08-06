# -*- coding: utf-8 -*-
"""LTC_Instructions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1F_EW9ib6KDiIMDHIFWcUH8Un52AXNo-7

# Liquid Time-Constant Neural Network (LTNN) Implementation
*Implemented by Michael B.Khani*
## Introduction
Liquid Time-Constant Neural Networks (LTNN) are a type of neural network designed to learn complex functions. Their primary characteristic is the unique processing in the hidden layer which relies on prior activations.

In this tutorial, we'll implement an LTNN to approximate the linear function y=2x+1. Our network will comprise of three layers:
1.   Input layer
2.   Hidden layer (with unique LTNN processing)
3.   Output layer

## Step 1: Problem Definition
We aim to approximate the function y=2x+1 using a set of input-output pairs as our training data. The goal is to fine-tune the network's parameters such that its predictions are as close as possible to the true outputs.

## Step 2: Data Preparation
To approximate our target function, we'll generate training data for the first five integers (0 to 4):
"""

#training_data = [(i, 2*i + 1) for i in range(5)]
training_data = [(0, 1), (1, 3), (2, 5), (3, 7), (4, 9)]

"""## Step 3: Network Architecture
1. Input Layer:
*   Single node for the input value I_0.
2. Hidden Layer:
*   Contains two nodes:
First node (x_0) always set to 1.0, acting as a bias.
Second node (x_1), processing input combined with the prior activation (x_0) using weights in theta and a constant A.
3. Output Layer:
*   A single node that produces the network's output. It computes the result using activations from the hidden layer combined with weight w and bias b.

## Step 4: Initialization
Before training, we must initialize the network parameters:
"""

import numpy as np # NumPy is a fundamental package for scientific computing with Python

theta = np.array([0.5, -0.2])
A = np.array([0.3])
tau = 1.0
w = 0.4
b = -0.1
learning_rate = 0.1

"""## Step 5: Forward Propagation
The `forward_pass` function is used to compute the network's output for a given input. It processes data through the layers, applying appropriate weights and biases.
"""

def forward_pass(input_value):
    I_0 = input_value
    x_0 = 1.0  # Use 1.0 instead of 0.0 to avoid vanishing activations initially
    S_0 = I_0 * theta[0] * A * x_0 + x_0 * theta[1] * A * x_0
    dx_dt_0 = x_0/tau + S_0
    x_1 = x_0 + dx_dt_0
    y_prime = x_1 * w + b
    return y_prime

"""## Step 6: Loss Calculation
We measure how close the network's predictions are to the true outputs using the Mean Squared Error (MSE) loss.


"""

def calculate_loss(y_prime, desired_output):
    return (y_prime - desired_output) ** 2

"""## Step 7: Backward Propagation
To optimize the network, we need to adjust its parameters based on the error it makes. The `backward_pass` function computes the gradients required to adjust the weights and biases.
"""

def backward_pass(y_prime, desired_output, input_value, x_t):
    dL_dy_prime = 2 * (y_prime - desired_output)
    dL_dw = dL_dy_prime * x_t
    dL_db = dL_dy_prime
    dL_dtheta_0 = dL_dy_prime * w * input_value
    dL_dtheta_1 = dL_dy_prime * w * x_t

    return dL_dw, dL_db, dL_dtheta_0, dL_dtheta_1

"""## Step 8: Accuracy Evaluation
Accuracy quantifies how well our model is performing. We want to know the percentage of predictions that are within a 5% margin of the true outputs:
"""

def calculate_accuracy():
    accurate_data_points = 0
    for input_value, desired_output in training_data:
        y_prime = forward_pass(input_value)
        percentage_error = abs((y_prime - desired_output) / desired_output) * 100
        if percentage_error < 5:
            accurate_data_points += 1
    return (accurate_data_points / len(training_data)) * 100

"""## Step 9: Training the LTC NN
Now, we'll iteratively adjust our network's parameters using the training data until our model reaches an accuracy of over 95%:
"""

# Training loop
while calculate_accuracy() < 95:
    total_loss = 0
    for input_value, desired_output in training_data:
        y_prime = forward_pass(input_value)
        loss = calculate_loss(y_prime, desired_output)
        total_loss += loss

        dL_dw, dL_db, dL_dtheta_0, dL_dtheta_1 = backward_pass(y_prime, desired_output, input_value, 1.0)  # x_t=1.0

        w -= learning_rate * dL_dw
        b -= learning_rate * dL_db
        theta[0] -= learning_rate * dL_dtheta_0
        theta[1] -= learning_rate * dL_dtheta_1

    print(f"Loss = {total_loss}, Accuracy = {calculate_accuracy()}%")

"""## Step 10: Testing the LTNN
Finally, once the network is trained, we can use it to make predictions and compare them to the actual outputs:


"""

# Test
print(f"\nTraining Data: {training_data}")
for input_value, _ in training_data:
    print(f"Input: {input_value}, Predicted Output: {forward_pass(input_value)}, Actual Output: {2*(input_value)+1}")

"""## Conclusion
You've now implemented a simple Liquid Time-Constant Neural Network (LTC NN) from scratch and trained it to approximate a linear function. While this example is elementary, the principles applied here can be expanded for more complex tasks.


---
> Note: Liquid Time-constant Neural networks (LTC NNs) are an interesting architecture that has the advantage of being relatively insensitive to gap length, compared to other RNNs, hidden Markov models, and other sequence learning methods1. While LTC NNs may not be as common as other architectures like CNNs or RNNs, it is always important to consider the nature and requirements of the problem at hand before choosing a suitable neural network architecture.
---


To get a full understanding, run the provided Python code and analyze the outputs. Experiment by changing parameters, training data, or network structure to gain deeper insights.
"""