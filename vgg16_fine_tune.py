# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras import backend as K

from sklearn.metrics import log_loss

from load_cifar10 import load_cifar10_data

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
    """VGG 16 Model for Keras
    Model Schema is based on
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    ImageNet Pretrained Weights
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of categories for our classification task
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), data_format='channels_first', input_shape=(channel, img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering="th"))
    model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering="th"))
    model.add(MaxPooling2D((2, 2), dim_ordering="th", strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
    model.add(Convolution2D(128, 3, 3, activation='relu', dim_ordering="th"))
    model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
    model.add(Convolution2D(128, 3, 3, activation='relu', dim_ordering="th"))
    model.add(MaxPooling2D((2, 2), dim_ordering="th", strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
    model.add(Convolution2D(256, 3, 3, activation='relu', dim_ordering="th"))
    model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
    model.add(Convolution2D(256, 3, 3, activation='relu', dim_ordering="th"))
    model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
    model.add(Convolution2D(256, 3, 3, activation='relu', dim_ordering="th"))
    model.add(MaxPooling2D((2, 2), dim_ordering="th", strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering="th"))
    model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering="th"))
    model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering="th"))
    model.add(MaxPooling2D((2, 2), dim_ordering="th", strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering="th"))
    model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering="th"))
    model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering="th"))
    model.add(MaxPooling2D((2, 2), dim_ordering="th", strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    model.load_weights('../keras/vgg16_weights.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    #for layer in model.layers[:10]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':

    # # Example to fine-tune on 3000 samples from Cifar10
    #
    # img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 2
    # batch_size = 16
    # nb_epoch = 10
    #
    # # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    # X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
    #
    # # Load our model
    model = vgg16_model(img_height, img_width, channel, num_classes)
    #
    # # Start Fine-tuning
    # print(X_train.transpose(0, 3, 1, 2).shape)
    # print(Y_train.shape)
    # print(X_valid.shape)
    # print(Y_valid.shape)
    # model.fit(X_train.transpose(0, 3, 1, 2), Y_train,
    #           batch_size=batch_size,
    #           nb_epoch=nb_epoch,
    #           shuffle=True,
    #           verbose=1,
    #           validation_data=(X_valid.transpose(0, 3, 1, 2), Y_valid),
    #           )
    #
    # # Make predictions
    # predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
    #
    # # Cross-entropy loss score
    # score = log_loss(Y_valid, predictions_valid)
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    X_train, Y_train = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    X_valid, Y_valid = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        X_train.transpose(0, 3, 1, 2), Y_train,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=(X_valid.transpose(0, 3, 1, 2), Y_valid),
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('first_try.h5')

