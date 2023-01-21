# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

dataset = pd.DataFrame({"p": [0, 0, 1, 1], "q": [0, 1, 0, 1], "r": [0, 0, 0, 1]})

X = dataset.iloc[:, 0:2]
Y = dataset.iloc[:, 2]

classifier = Sequential()
classifier.add(Dense(2, activation="sigmoid", input_dim=2))
classifier.add(Dense(1, activation="sigmoid"))

classifier.compile(optimizer=SGD(learning_rate=0.01), loss="mean_squared_error", metrics=["accuracy"])

history = classifier.fit(X, Y, batch_size=1, epochs=10000, callbacks=[EarlyStopping(monitor="loss")])

eval_model = classifier.evaluate(X, Y)
print("-----")
print(eval_model)

y_pred = classifier.predict(X)
print(y_pred > 0.7)
