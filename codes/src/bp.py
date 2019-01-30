#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime
import src.network2

def backprop(x, y, biases, weights, cost, num_layers):
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    activation = x
    activations = [x]
    before_activations = []
    for bias, weight in zip(biases, weights):
        before_activation = np.dot(weight, activation)+bias
        before_activations.append(before_activation)
        activation = sigmoid(before_activation)
        activations.append(activation)

    delta = (cost).delta(activations[-1], y)

    for l in range(1, num_layers):
        if l != 1:
            delta = np.dot(weights[-l+1].transpose(), delta) * sigmoid_prime(before_activations[-l])
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)