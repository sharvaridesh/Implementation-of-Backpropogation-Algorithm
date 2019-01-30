#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = 'C:/Users/Aravind/Documents/NCSU/3rd sem/ECE 542/project/ece542-2018fall/project/data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')

    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 100, 10])
    model.gradient_check(training_data=train_data, layer_id=0, unit_id=6, weight_id=2)

def test():
    train_data, valid_data, test_data = load_data ()
    model = network2.load("weights.json")
    print("Accuracy on test data: ", model.accuracy(test_data)/100)

def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784, 100, 10])
    # train the network using SGD
    print(model)
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = model.SGD(
        training_data=train_data,
        epochs=100,
        mini_batch_size=128,
        eta=5e-3,
        lmbda = 0.0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
    plt.figure()
    plt.plot(training_cost)
    plt.xlabel("epochs")
    plt.ylabel("Cost")
    plt.title('Cost during training')
    plt.show()
    plt.figure()
    plt.plot(training_accuracy)
    plt.title('Accuracy during training')
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.show()
    plt.figure()
    plt.plot(evaluation_cost)
    plt.title('Cost during validation')
    plt.xlabel("epochs")
    plt.ylabel("Cost")
    plt.show()
    plt.figure()
    plt.plot(evaluation_accuracy)
    plt.title('Accuracy during validation')
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.show()
    model.save("weights.json")

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
    if FLAGS.test:
        test()