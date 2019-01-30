#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: activation.py

import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))
