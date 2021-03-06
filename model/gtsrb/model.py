import csv
import numpy
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from callback.remote_monitor import KeruviRemoteMonitor

IMG_WIDTH = 32
IMG_HEIGHT = 32
NUM_CLASSES = 43

# Read CSV file containing a mapping of traffic
# sign ID to its name and convert it to a dictionary.
ID_TO_NAME_INDEX = {}
with open(os.path.join('gtsrb', 'data', 'traffic_sign_names.csv')) as csv_fd:
    csv_reader = csv.reader(csv_fd, delimiter=',')
    # Skip title row.
    next(csv_reader)
    for row in csv_reader:
        class_id = int(row[0])
        class_name = row[1]
        ID_TO_NAME_INDEX[class_id] = class_name

def get_road_sign_name(road_sign_id):
    return ID_TO_NAME_INDEX[road_sign_id]

def preprocess_image(image):
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    return numpy.asarray(img)

def load_gtsrb_data(path):
    images = []
    labels = []
    for class_id in range(NUM_CLASSES):
        class_folder_name = format(class_id, '05d')
        class_csv_file_name = 'GT-{}.csv'.format(class_folder_name)
        class_csv_file_path = os.path.join(path, 'metadata', class_csv_file_name)
        with open(class_csv_file_path, 'r') as csv_fd:
            # Read CSV file with metadata for
            # each dataset entry of current class.
            csv_reader = csv.reader(csv_fd, delimiter=';')
            # Skip title row.
            next(csv_reader)
            for row in csv_reader:
                img_name = row[0]
                img_label = row[7]
                img_file_path = os.path.join(path, 'images', class_folder_name, img_name)
                img_content = Image.open(img_file_path)
                img_matrix = preprocess_image(img_content)
                images.append(img_matrix)
                labels.append(to_categorical(img_label, num_classes=NUM_CLASSES))
    return numpy.array(images) / 255, numpy.array(labels)

def load_model():
    # Define model (convolutional neural network).
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.BatchNormalization(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT,3)))
    model.add(tf.keras.layers.Conv2D(64, (5,5), strides=(2,2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (5,5), strides=(2,2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    # Build model.
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=['accuracy']
    )

    return model

def run(root_url):
    # Create data generator serving training
    # data directly from file system directory
    # transforming it randomly on the fly.
    train_data_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
        validation_split=0.2
    )
    train_data = train_data_generator.flow_from_directory(
        os.path.join('gtsrb', 'data', 'train', 'images'),
        target_size=(32, 32),
        batch_size=128
    )

    # Prepare and build model.
    model = load_model()

    # Train model using custom Keras callback.
    monitor = KeruviRemoteMonitor(root=root_url, model_id='gtsrb', use_batch_callback=True)
    model.fit(
        train_data,
        epochs=40,
        callbacks=[monitor],
        verbose=0
    )

    # Evaluate model.
    test_imgs, test_labels = load_gtsrb_data(os.path.join('gtsrb', 'data', 'test'))
    metrics = model.evaluate(test_imgs, test_labels, verbose=0)
    print(metrics)

    # Save fully trained model.
    model.save('gtsrb_model.h5')