import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def forward(X,w):
  return sigmoid(np.matmul(X,w))

def gradient(X,Y,w):
  error=forward(X,w) - Y
  return np.matmul(X.T, (error)) / X.shape[0]

def train(X,Y, iterations, lr):
  w=np.zeros((X.shape[1], 1), dtype=np.float64)
  for i in range(iterations):
    print(f"iteration {i} => loss {mse_loss(X,Y,w)}")
    w -= gradient(X, Y, w) * lr
  return w

def mse_loss(X,Y,w):
  return np.average((forward(X,w) - Y) ** 2)
  
def sigmoid(z):
  return 1/ (1 + np.exp(-z))

def classify(X,w):
  return np.round(forward(X,w))

def test(X,Y,w):
  total_examples = X.shape[0]
  correct_results = np.sum(classify(X,w) == Y)
  success_pct = correct_results * 100 / total_examples
  print(f"Success: {correct_results}/{total_examples} {success_pct}%")
