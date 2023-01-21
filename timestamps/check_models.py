import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam

# from sklearn.preprocessing import MinMaxScaler


x_len = 168*2
y_len = 168

df = pd.read_csv("data.csv")

df['time'] = pd.to_datetime(df['time'])
df['hour'] = df['time'].map(lambda x: x.hour/24)


def plot_dataframe(df, x_col='time', y_cols=None, title=None):
    if y_cols is None:
        y_cols = df.columns.drop('time')
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    plt.figure(figsize=(16, 9))

    if title is not None:
        plt.title(title)

    for col in y_cols:
        plt.plot(df[x_col], df[col], label=col)
    plt.legend()
    plt.show()


def get_windows(df: pd.DataFrame, sizeX=168, sizeY=24):
    startX = 0
    endX = startX + sizeX
    startY = endX
    endY = startY + sizeY
    Xs = []
    Ys = []

    while endY < len(df):
        x = df.iloc[startX: endX, 1:].values
        y = df.iloc[startY: endY, 1].values
        Xs.append(x)
        Ys.append(y)

        startX += 1
        endX = startX + sizeX
        startY = endX
        endY = startY + sizeY

    return np.array(Xs), np.array(Ys)


def gen_next_seq(seq, last, size):
    res = []
    for i in range(last, last+size):
        res.append(seq[i % len(seq)])
    return res, last+size % len(seq)


train_last = int(len(df)*0.6)
valid_last = int(len(df)*0.8)

train_df = df.iloc[:train_last]
valid_df = df.iloc[train_last:valid_last]
test_df = df.iloc[valid_last:]

# gbps_scaler = MinMaxScaler()
# train_df['Gbps'] = gbps_scaler.fit_transform(train_df['Gbps'].values.reshape(-1, 1))
# valid_df['Gbps'] = gbps_scaler.transform(valid_df['Gbps'].values.reshape(-1, 1))
# test_df['Gbps'] = gbps_scaler.transform(test_df['Gbps'].values.reshape(-1, 1))


Xtrain, Ytrain = get_windows(train_df, sizeX=x_len, sizeY=y_len)
Xvalid, Yvalid = get_windows(valid_df, sizeX=x_len, sizeY=y_len)
Xtest, Ytest = get_windows(test_df, sizeX=x_len, sizeY=y_len)

model = Sequential()

model.add(LSTM(min(x_len*2, 128), input_shape=(168, 2), return_sequences=True))
model.add(LSTM(min(y_len, 64), return_sequences=False))
model.add(Dense(y_len, activation='linear'))
model.compile(optimizer=Adam(learning_rate=0.004), loss='mse', metrics=['mse'])

model.load_weights('model_weights.h5')

hours_seq = [i/24 for i in range(24)]
last_hour = (df['time'][0].weekday() * 24 + valid_last % y_len) % len(hours_seq)


y_pred_seq = []
number_of_predictions = Xtest.shape[0] // y_len
hours = []


for i in range(number_of_predictions):
    if not i:
        X = Xtest[0]
    else:
        last_predicted = y_pred_seq[-1]
        hours, last_hour = gen_next_seq(hours_seq, last_hour, y_len)
        X_append = np.array([last_predicted, hours]).T
        X = np.concatenate((X, X_append))[y_len:]
    y_pred = np.array(model.predict(X.reshape(1, *X.shape)))
    y_pred = y_pred.flatten()
    y_pred_seq.append(y_pred)

y_pred_seq = list(np.array(y_pred_seq).flatten())

for _ in range(x_len):
    y_pred_seq.insert(0, np.NaN)

for _ in range(len(Ytest) - len(y_pred_seq) + y_len + x_len):
    y_pred_seq.append(np.NaN)

test_df = test_df.dropna()
test_df['prediction'] = y_pred_seq

plot_dataframe(test_df, y_cols=['Gbps', 'prediction'])

res = []
for i in list(test_df['Gbps'].index):
    if not np.isnan(test_df['prediction'][i]):
        res.append((test_df['Gbps'][i] - test_df['prediction'][i]) / test_df['Gbps'][i])

with open('model_weights_preds.json', 'w') as file:
    json.dump(res, file, indent=4)
