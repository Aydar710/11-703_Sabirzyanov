import os
import time

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import logging
import numpy as np
import pygame


def draw_and_save_number():
    # # Window size
    WIDTH = 28
    HEIGHT = 28
    # WINDOW_WIDTH = 420
    # WINDOW_HEIGHT = 420
    # FPS = 60
    # PIXELS_WIDTH = 28  # How many big-pixels vertically
    # PIXELS_HEIGHT = 28  # How many big-pixels horizontally
    #
    # # background & colours
    FILL = (0, 0, 0)
    COLOR = 'white'
    CAPTION = 'Draw a number'
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(CAPTION)
    line_start = None
    screen.fill(FILL)

    while True:
        mouse_pos = pygame.mouse.get_pos()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                break
            if e.type == pygame.MOUSEBUTTONUP:
                line_start = None if line_start else mouse_pos
            if e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 2:
                    screen.fill((0, 0, 0))
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_s:
                    pygame.image.save(screen, 'digit_2.png')
                    print('Saved png file')
        else:
            if line_start:
                pygame.draw.line(screen, pygame.color.Color(COLOR), line_start, mouse_pos, width=3)
                line_start = mouse_pos
            pygame.display.flip()
            continue
        break
    return


tf.get_logger().setLevel(logging.ERROR)


# Making CNN model
def mnist_cnn_model():
    image_size = 28
    num_channels = 1
    # 1 for grayscale images
    num_classes = 10

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                     input_shape=(image_size, image_size, num_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def mnist_cnn_train(model):
    (train_digits, train_labels), (test_digits, test_labels) = keras.datasets.mnist.load_data()

    # Get image size
    image_size = 28
    num_channels = 1
    # 1 for grayscale images
    # re-shape and re-scale the images data

    train_data = np.reshape(train_digits, (train_digits.shape[0], image_size, image_size, num_channels))
    train_data = train_data.astype('float32') / 255.0
    # encode the labels - we have 10 output classes
    num_classes = 10
    train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)

    # re-shape and re-scale the images validation data
    val_data = np.reshape(test_digits, (test_digits.shape[0], image_size, image_size, num_channels))
    val_data = val_data.astype('float32') / 255.0

    # encode the labels - have 10 output classes
    val_labels_cat = keras.utils.to_categorical(test_labels, num_classes)
    print("Training the network...")
    t_start = time.time()

    # Training
    model.fit(train_data, train_labels_cat, epochs=20, batch_size=64, validation_data=(val_data, val_labels_cat))
    print("Done, dT:", time.time() - t_start)

    return model


def cnn_predict_digits(model, image_file):
    image_size = 28
    img = keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size), color_mode='grayscale')

    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    img_arr = img_arr.reshape((1, 28, 28, 1))

    result = model.predict_classes([img_arr])
    return result[0]


if __name__ == '__main__':
    # Draw digit
    # draw_and_save_number()
    # Creating model
    # model = mnist_cnn_model()
    # Training model
    # mnist_cnn_train(model)
    # Saving model
    # model.save('cnn_digits_28x28.h5')
    # Load trained model
    model = tf.keras.models.load_model('cnn_digits_28x28.h5')

    # Testing
    print(cnn_predict_digits(model, 'digit_1.png'))
    print(cnn_predict_digits(model, 'digit_2.png'))
    print(cnn_predict_digits(model, 'digit_3.png'))
    print(cnn_predict_digits(model, 'digit_4.png'))
    print(cnn_predict_digits(model, 'digit_7.png'))
