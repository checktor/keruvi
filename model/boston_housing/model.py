from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from callback.remote_monitor import KeruviRemoteMonitor

def run(root_url):
    # Get Boston Housing dataset.
    (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

    # Define model (neural network).
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=train_data.shape[1:]))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    # Build model.
    rmsprop = RMSprop(learning_rate=0.01)
    model.compile(optimizer=rmsprop, loss='mse')

    # Train model using custom Keras callback.
    monitor = KeruviRemoteMonitor(root=root_url, model_id='boston')
    model.fit(
        train_data,
        train_labels,
        epochs=200,
        batch_size=16,
        validation_split=0.2,
        callbacks=[monitor],
        verbose=0
    )
    
    # Evaluate model.
    metrics = model.evaluate(test_data, test_labels, verbose=0)
    print(metrics)

    # Save fully trained model.
    model.save('boston_housing_model.h5')