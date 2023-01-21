from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

data_train_path = "X:\\Informatyczne\\python\\pycharmprojects\\UMCS.AI\\CNN_dogs_cats\\data\\train"

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.25,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    directory=data_train_path,
    target_size=(256, 256),
    color_mode='rgb',
    batch_size=32,
    class_mode='binary',
    seed=2021
)

model = Sequential()

model.add(Conv2D(filters=10, kernel_size=(5, 5), padding='same', input_shape=(256, 256, 3)))
model.add(Conv2D(filters=10, kernel_size=(5, 5), padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(filters=20, kernel_size=(5, 5), padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(filters=30, kernel_size=(4, 4), padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(filters=40, kernel_size=(3, 3), padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(filters=50, kernel_size=(3, 3), padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(filters=100, kernel_size=(3, 3), padding='same'))

model.add(Flatten())

model.add(Dense(300, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(train_gen, batch_size=32, epochs=1000,
                    callbacks=[EarlyStopping(monitor='acc'), ModelCheckpoint('best_model2.h5')],
                    max_queue_size=128, workers=20)

model.save('model2.h5')



