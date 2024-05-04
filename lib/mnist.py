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

def one_hot_encode(Y, exact_match=False, n_classes=10):
  n_labels = Y.shape[0]
  encoded_Y = np.zeros((n_labels,n_classes))

  value_to_index = {}
  if exact_match:
    value_to_index, _ignore = one_hot_encoding_values(Y)
    
  for i in range(n_labels):
    label = Y[i]
    if exact_match:
      label = value_to_index.get(label)
    encoded_Y[i][label] = 1
  return encoded_Y

def one_hot_encoding_values(Y):
  unique_values = np.unique(Y)
  value_to_index = { v: ord(v) % len(unique_values)  for v in unique_values }
  index_to_value = { v: k  for k,v in value_to_index.items() }
  return value_to_index,index_to_value

def one_hot_encoding_value_from_index(Y, i):
  _ignore,index_to_value=one_hot_encoding_values(Y)
  return index_to_value.get(i)

def extract_test_data(data_np, test_size=0.23):
  test_size = round(data_np.shape[0] * test_size)
  print(f"test_size: {test_size}")
  X_test = data_np[:test_size]
  X_train = data_np[test_size:]
  print(X_test)
  Y_train = X_train[:,-1]
  Y_test = X_test[:,-1]
  X_train = X_train[:,:-1]
  X_test = X_test[:,:-1]
  return X_train,Y_train, X_test,Y_test

