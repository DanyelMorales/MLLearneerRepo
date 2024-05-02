import numpy as np
import gzip
import struct
import requests

def load_images(filename):
  with gzip.open(filename, 'rb') as ff:
    # read the header information into a bunch of variables
    _ignored, n_images, columns, rows = struct.unpack('>IIII', ff.read(16))
    # read all the pixels into a numpy array bytes
    all_pixels = np.frombuffer(ff.read(), dtype=np.uint8)
    #reshape the pixels into a matrix where each line is an image
    return all_pixels.reshape(n_images, columns * rows)

def prepend_bias(X):
  # ("axis=1" stands for: "insert a column, not a row")
  return np.insert(X,0,1, axis=1)


def load_labels(filename):
  with gzip.open(filename, 'rb') as f:
    # skip header bytes
    f.read(8)
    # Read all the labels into a list:
    all_labels = f.read()
    # Reshape the list of labels into one column matrix:
    return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1,1)


def encode_fives(Y):
  # convert all 5s to 1, and everthing else to 0
  return (Y==1).astype(int)

def one_hot_encode(Y, n_classes=10):
  n_labels = Y.shape[0]
  encoded_Y = np.zeros((n_labels,n_classes))
  for i in range(n_labels):
    label = Y[i]
    encoded_Y[i][label] = 1

  return encoded_Y


