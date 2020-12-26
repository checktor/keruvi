import numpy
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from callback.remote_monitor import KeruviRemoteMonitor

CHARS74K_TRAIN_DIRECTORY = os.path.join('handwritten_digits', 'data', 'train')
CHARS74K_TEST_DIRECTORY = os.path.join('handwritten_digits', 'data', 'test')

def load_mnist_data():
    # Get and prepare MNIST data.
    (train_imgs, train_labels), (test_imgs, test_labels) = tf.keras.datasets.mnist.load_data()

    # Training images (60,000 images, grayscale, 28 x 28).
    train_imgs = train_imgs.reshape((-1, 28, 28, 1))
    # Normalize pixels.
    train_imgs = train_imgs.astype(float)
    train_imgs /= 255
    # Training labels (one-hot encoding, 10 categories).
    train_labels = to_categorical(train_labels)

    # Testing images (10,000 images, grayscale, 28 x 28)
    test_imgs = test_imgs.reshape((-1, 28, 28, 1))
    # Normalize pixels.
    test_imgs = test_imgs.astype(float)
    test_imgs /= 255
    # Testing labels (one-hot encoding, 10 categories).
    test_labels = to_categorical(test_labels)

    return (train_imgs, train_labels), (test_imgs, test_labels)

def _load_chars74k_data(path):
    imgs = []
    labels = []
    for img_directory in sorted(os.listdir(path)):
        # Training data is stored in directories with "digit_" prefix
        # followed by corresponding class label, i.e. "digit_2
        # directory contains all examples for digit "2" class.
        if img_directory.startswith('digit_'):
            for img_file in sorted(os.listdir(os.path.join(path, img_directory))):
                # Read image from file system.
                img_raw = Image.open(os.path.join(path, img_directory, img_file))
                # Convert to grayscale.
                img_raw = img_raw.convert('L')
                # Resize to 28 x 28 pixels.
                img_raw = img_raw.resize((28, 28))
                # Convert image to array.
                img = tf.keras.preprocessing.image.img_to_array(img_raw)
                # Normalize pixels.
                img.astype(float)
                img /= 255
                imgs.append(img)
                # Extract corresponding label from folder name.
                label_raw = int(img_directory.split('_')[1])
                # Convert to one-hot encoding (10 categories).
                label = to_categorical(label_raw, 10)
                labels.append(label)
    return numpy.array(imgs), numpy.array(labels)
    
def load_chars74k_data():
    # Load training data.
    train_imgs, train_labels = _load_chars74k_data(CHARS74K_TRAIN_DIRECTORY)
    # Load testing data.
    test_imgs, test_labels = _load_chars74k_data(CHARS74K_TEST_DIRECTORY)

    return (train_imgs, train_labels), (test_imgs, test_labels)

def run(root_url):
    # Get MNIST data.
    (mnist_train_imgs, mnist_train_labels), (mnist_test_imgs, mnist_test_labels) = load_mnist_data()

    # Get Chars74K data.
    (chars74k_train_imgs, chars74k_train_labels), (chars74k_test_imgs, chars74k_test_labels) = load_chars74k_data()

    # Combine both datasets to a single one.
    train_imgs = numpy.concatenate((mnist_train_imgs, chars74k_train_imgs), axis=0)
    train_labels = numpy.concatenate((mnist_train_labels, chars74k_train_labels), axis=0)
    test_imgs = numpy.concatenate((mnist_test_imgs, chars74k_test_imgs), axis=0)
    test_labels = numpy.concatenate((mnist_test_labels, chars74k_test_labels), axis=0)

    # Create data generator to randomly
    # transform training data on the fly.
    train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
        validation_split=0.2
    )
    train_data_generator.fit(train_imgs)

    # Define model (convolutional neural network).
    model = Sequential()

    # Convolutional layer 1.
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=train_imgs.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Convolutional layer 2.
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Dropout.
    model.add(Dropout(0.5))

    model.add(Flatten())

    # Dense layer 1.
    model.add(Dense(128, activation='relu'))

    # Dense layer 2.
    model.add(Dense(64, activation='relu'))

    # Dense layer 3.
    model.add(Dense(10, activation='softmax'))

    # Build model.
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # Train model using custom Keras callback.
    monitor = KeruviRemoteMonitor(root=root_url, model_id="digits", use_batch_callback=True)
    model.fit(
        train_data_generator.flow(train_imgs, train_labels, batch_size=256),
        epochs=5,
        callbacks=[monitor],
        verbose=0
    )
    
    # Evaluate model.
    metrics = model.evaluate(test_imgs, test_labels, verbose=0)
    print(metrics)

    # Save fully trained model.
    model.save('digits_model.h5')