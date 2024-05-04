import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import mnist

def forward(X,w):
  return sigmoid(np.matmul(X,w))

def gradient(X,Y,w):
  error=forward(X,w) - Y
  return np.matmul(X.T, (error)) / X.shape[0]

def train(X_train, Y_train, X_test, Y_test, iterations, lr):
  w=np.zeros((X_train.shape[1], Y_train.shape[1]), dtype=np.float64)
  for i in range(iterations):
    report(i, X_train, Y_train, X_test, Y_test, w)
    w -= gradient(X_train, Y_train, w) * lr
  report(iterations, X_train, Y_train, X_test, Y_test, w)
  return w

def loss(X,Y,w):
  y_hat = forward(X,w)
  first_term = Y * np.log(y_hat)
  second_term = ( 1 - Y ) * np.log(1-y_hat)
  return -np.sum(first_term + second_term)/X.shape[0]
  
def sigmoid(z):
  return 1/ (1 + np.exp(-z))

def classify(X,w, from_one_hot_encode=False):
  y_hat = forward(X,w)
  labels=np.argmax(y_hat, axis=1)
  return labels.reshape(-1,1)

def report(iteration, X_train, Y_train, X_test, Y_test, w, from_one_hot_encode=False):  
  classified = classify(X_test, w)                                                  
  if from_one_hot_encode:
    print(y_hat)
    classified=mnist.one_hot_encoding_value_from_index(y_hat,classified[0])
  matches = np.count_nonzero(classified == Y_test)
  n_test_examples = Y_test.shape[0]
  matches = matches * 100.0 / n_test_examples
  training_loss = loss(X_train, Y_train, w)
  print(f"{iteration} - Loss: {training_loss}, \n {matches}%")
  
def test(X,Y,w):
  total_examples = X.shape[0]
  correct_results = np.sum(classify(X,w) == Y)
  success_pct = correct_results * 100 / total_examples
  print(f"Success: {correct_results}/{total_examples} {success_pct}%")
