import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix
tf.device('gpu')
dataset = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1], "a": [0, 0, 0, 1]})

X = dataset.iloc[:, 0:2]
Y = dataset.iloc[:, 2]
print(X, Y)

classified = Sequential()

classified.add(Dense(2, activation='sigmoid', input_dim=2))
classified.add(Dense(1, activation='sigmoid'))

classified.compile(optimizer=SGD(learning_rate=0.1), loss='mean_squared_error', metrics=['accuracy'])

history = classified.fit(X, Y, batch_size=4, epochs=10000, use_multiprocessing=False)
eval_model = classified.evaluate(X, Y)
print(eval_model)

y_pred = classified.predict(X)
y_pred_final = y_pred > 0.7

cm = confusion_matrix(Y, y_pred_final)
