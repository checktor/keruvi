# keruvi (Keras runtime visualisation)

Runtime visualisation of TensorFlow Keras model training with Node.js.

## Requirements

* TensorFlow (https://www.tensorflow.org/)
    * deep neural networks
* Node.js (https://nodejs.org/en/)
    * JavaScript runtime

The example models in [model](model/) directory may require further packages, i.e.:

* NumPy (https://numpy.org/)
    * numerical computations
* Pillow (https://python-pillow.org/)
    * image processing

## Install

TensorFlow libraries for Python can be installed via pip. If necessary, install python3 and python3-pip packages first.

    # sudo apt install python3
    # sudo apt install python3-pip
    pip3 install tensorflow

Node project dependencies (e.g. Express framework) can be installed via npm. If necessary, install Node.js and npm first. See https://nodejs.org/en/ and https://www.npmjs.com/ for details.

    # install Node.js and npm
    cd server/
    npm i --production

## Usage

During training of Keras models, metrics can be received via Keras callbacks. Use custom [KeruviRemoteMonitor](model/callback/remote_monitor.py) callback to send runtime data to a Node server which is able to visualize current training status. Start this server as follows:

    cd server/
    npm run start

By default, this server will be accessible through port 3000. Use environment variable `KERUVI_PORT` to change this behavior.

Connect specific Keras model to visualisation server using `KeruviRemoteMonitor`. In the following example, the callback instance is configured with server URL `http://localhost:3000` and ID `boston`:

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import RMSprop

    from callback.remote_monitor import KeruviRemoteMonitor

    # Create model.
    model = Sequential()

    # Add Keras layers as desired.
    # ...

    # Build model.
    model.compile(optimizer=RMSprop(learning_rate=0.01), loss='mse')

    # Train model using custom Keras callback.
    monitor = KeruviRemoteMonitor(root='http://localhost:3000', model_id='boston')
    model.fit(
        your_training_data,
        your_training_labels,
        epochs=200,
        batch_size=16,
        validation_split=0.2,
        callbacks=[monitor],
        verbose=0
    )

To get corresponding training metrics, e.g. for each epoch (`on_epoch_end` callback functionality), visit URL `http://localhost:3000/epoch/boston`. This endpoint is configured to be used as an EventSource:

    let oEventSource = new EventSource('epoch/boston');
    oEventSource.onmessage = function(oEvent) {

        // Update visualisation.
        // ...

    }

See https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback for more informationen concerning Keras callbacks. Sample models using a custom Keras callback can be found in [model](model/) folder. Run corresponding `_example.py` scripts for testing, e.g. [boston_housing_example.py](model/boston_housing_example.py) to train a model using the Boston Housing dataset. Currently, trainable example models for the following datasets are provided:

* [Boston Housing](model/boston_housing) (prediction of house prices)
* [GTSRG](model/gtsrb) (classification of traffic sign images)
* [MNIST and Chars74K](model/handwritten_digits) (classification of handwritten digits)
* [IMDb](model/imdb) (sentiment analysis)