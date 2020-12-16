from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from callback.remote_monitor import KeruviRemoteMonitor

def run(root_url):
    # Get and prepare MNIST data.
    (train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data() 
    
    # Training images (grayscale, 28 x 28).
    train_imgs = train_imgs.reshape((60000, 28, 28, 1))
    train_imgs = train_imgs.astype(float)
    train_imgs /= 255
    # Training labels (one-hot encoding, 10 categories).
    train_labels = to_categorical(train_labels)

    # Testing images (grayscale, 28 x 28)
    test_imgs = test_imgs.reshape((10000, 28, 28, 1))
    test_imgs = test_imgs.astype(float)
    test_imgs /= 255
    # Testing labels (one-hot encoding, 10 categories).
    test_labels = to_categorical(test_labels)

    # Define model (convolutional neural network).
    model = Sequential()

    # Convolutional layer 1.
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=train_imgs.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolutional layer 2.
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

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
    monitor = KeruviRemoteMonitor(root=root_url, model_id="mnist", use_batch_callback=True)
    model.fit(
        train_imgs,
        train_labels,
        batch_size=256,
        epochs=5,
        validation_split=0.2,
        callbacks=[monitor],
        verbose=0
    )
    
    # Evaluate model.
    metrics = model.evaluate(test_imgs, test_labels, verbose=0)
    print(metrics)

    # Save fully trained model.
    model.save('mnist_model.h5')