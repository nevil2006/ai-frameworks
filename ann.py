import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.datasets import mnist

# Load built-in dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Loaded successfully")