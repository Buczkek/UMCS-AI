import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


# from sklearn.model_selection import TimeSeriesSplit

def targets(values, y_len):
    windows = []
    win_start = 0
    win_end = y_len

    while win_end < len(values):
        windows.append(values[win_start:win_end])
        win_start += 1
        win_end = win_start + y_len
    return windows


df = pd.read_csv("data.csv")
df['time'] = pd.to_datetime(df['time'])
df['Gbps'] = df['Gbps'].astype(np.float64)
df['hour'] = df['time'].map(lambda x: x.hour) / 24
xlen = 168 * 2
ylen = 168
train_end = int(len(df) * .6)
valid_end = int(len(df) * .8)
train_df = df.iloc[:train_end]
valid_df = df.iloc[train_end:valid_end]
test_df = df.iloc[valid_end:]
scaler = MinMaxScaler()
train_df['Gbps'] = scaler.fit_transform(train_df['Gbps'].values.reshape(-1, 1))
valid_df['Gbps'] = scaler.transform(valid_df['Gbps'].values.reshape(-1, 1))
test_df['Gbps'] = scaler.transform(test_df['Gbps'].values.reshape(-1, 1))
train_gen = TimeseriesGenerator(train_df.iloc[:-ylen, 1:].values, targets(train_df.iloc[:, 1], ylen), length=xlen)
valid_gen = TimeseriesGenerator(valid_df.iloc[:-ylen, 1:].values, targets(valid_df.iloc[:, 1], ylen), length=xlen)
model = Sequential()
model.add(LSTM(min(xlen * 2, 128), input_shape=(xlen, 2), return_sequences=True))
model.add(LSTM(min(ylen, 64), return_sequences=False))
model.add(Dense(ylen, activation='linear'))
model.compile(optimizer=Adam(lr=0.004), loss='mse', metrics=['mse'])
stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
model.fit(train_gen, validation_data=valid_gen, epochs=1000, callbacks=stop)