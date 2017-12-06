from glob import glob
import numpy as np
from keras.backend import set_image_dim_ordering
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.optimizers import adam


def normalize(x):
    x /= 255
    return x


def load_dataset(type):
    if type is 'train' or type is 'validation' or type is 'test':
        path = 'data/'
        path += type
        path += '/*.jpg'
        set_image_dim_ordering('th')
        x = []
        y = []
        for img_path in glob(path):
            img = load_img(img_path, grayscale=True)
            y_post = img_path.find('_') + 1
            x.append(img_to_array(img))
            y.append(img_path[y_post])
        x = np.array(x)
        y = np.array(y)
        x = normalize(x)
        y = to_categorical(y, 7)
        return x, y
    else:
        print('type is invalid, please use "train", "validation", or "test"')


def train(x, y, x_val=None, y_val=None, save_model=False, lr=1e-3, epoch=50, rotation_range=0.0, width_shift_range=0.0,
          height_shift_range=0.0,
          horizontal_flip=True, vertical_flip=False):

    model = Sequential()
    # Conv Layer 1 (depth 12, ukuran filter 3 x 3)  - MaxPool 1
    model.add(Conv2D(12, (3, 3), padding='same', input_shape=x.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Conv Layer 2 (depth 24, ukuran filter 3 x 3) - MaxPool 2
    model.add(Conv2D(24, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Conv Layer 3 (depth 48, ukuran filter 3 x 3) - MaxPool 3
    model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Conv Layer 4 (depth 96, ukuran filter 3 x 3) - MaxPool 4
    model.add(Conv2D(96, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())

    # Fully Connected 1
    model.add(Dense(960, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Fully Connected 2
    model.add(Dense(480, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Sofmax Classifier
    model.add(Dense(7, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=adam(lr=lr), metrics=['accuracy'])

    # Data augmentation
    dataGenerator = ImageDataGenerator(rotation_range=rotation_range, width_shift_range=width_shift_range,
                                       height_shift_range=height_shift_range, horizontal_flip=horizontal_flip,
                                       vertical_flip=vertical_flip)
    dataGenerator.fit(x)

    # Train dan Validasi
    model.fit_generator(dataGenerator.flow(x, y, batch_size=128), validation_data=(x_val, y_val),
                        epochs=epoch, workers=10, verbose=2)

    # Save model dan weight
    if save_model:
        model.save('model_final_fix.h5')
        model.save_weights('weight_model_final_fix.h5')

    # Evaluasi model
    scores = model.evaluate(x, y, verbose=0)
    print('--- Result---')
    scores = model.evaluate(x_val, y_val, verbose=0)
    print('Validation loss: %.4f' % scores[0])
    print('Validation accuracy: %.3f%%' % (scores[1] * 100))
