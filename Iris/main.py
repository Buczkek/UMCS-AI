import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

start = time.time()

tf.device('gpu')
dataset = pd.read_csv("iris.csv")

X = dataset.iloc[:, 0:4].values
Y = dataset.iloc[:, 4].values
# print(X, Y)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

encoder = LabelEncoder()
y1 = encoder.fit_transform(Y)
# print(X, y1)

Y = pd.get_dummies(y1).values

# print(Y)


Xtrain, Xtest, Ytrain, Ttest = train_test_split(X, Y, test_size=0.2, shuffle=True)

classified = Sequential()

classified.add(Dense(30, activation='relu', input_dim=4))
classified.add(Dense(10, activation='relu'))
classified.add(Dense(5, activation='relu'))
classified.add(Dense(3, activation='softmax'))

classified.compile(optimizer=SGD(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

history = classified.fit(Xtrain, Ytrain, batch_size=4, epochs=30, use_multiprocessing=True, )
eval_model = classified.evaluate(Xtrain, Ytrain)
print(eval_model)

y_pred = classified.predict(Xtest)

# cm = confusion_matrix(Y, y_pred_final)


# pca rfe

print(time.time()-start)
