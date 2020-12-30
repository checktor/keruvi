import numpy
import string
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence

from callback.remote_monitor import KeruviRemoteMonitor

VOCABULARY_SIZE = 20000
PAD_MAX_LENGTH = 2500
PAD_WORD_ID = 0
START_WORD_ID = 1
UNKNOWN_WORD_ID = 2
INDEX_FROM = 3

# Get IMDb index (word -> ID).
WORD_TO_ID_INDEX = imdb.get_word_index()
WORD_TO_ID_INDEX = { key: value + INDEX_FROM for key, value in WORD_TO_ID_INDEX.items() }
WORD_TO_ID_INDEX['<PAD>'] = PAD_WORD_ID
WORD_TO_ID_INDEX['<START>'] = START_WORD_ID
WORD_TO_ID_INDEX['<UNKNOWN>'] = UNKNOWN_WORD_ID

# Get IMDb index (ID -> word).
ID_TO_WORD_INDEX = { value: key for key, value in WORD_TO_ID_INDEX.items() }

def convertIdListToString(id_list):
    # Replace each word ID in provided
    # list with its string representation.
    word_list = [ID_TO_WORD_INDEX[id] for id in id_list]
    # Concatenate all words in converted list to
    # a single string separated by white spaces.
    word_string = ' '.join(word_list)
    return word_string

def convertStringToIdList(sentence):
    # Convert provided string to lowercase.
    sentence = sentence.lower()
    # Remove all sorts of punctuation.
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # Convert retrieved string to a list of
    # words using a white space as delimiter.
    # Then replace each word with its ID.
    id_list = []
    for word in sentence.split(' '):
        id_list.append(WORD_TO_ID_INDEX[word])
    # Pad word ID list to uniform length.
    id_list = sequence.pad_sequences([id_list], maxlen=PAD_MAX_LENGTH, value=PAD_WORD_ID)
    return numpy.array(id_list)

def run(root_url):
    # Get IMDb dataset.
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=VOCABULARY_SIZE,
        start_char=START_WORD_ID,
        oov_char=UNKNOWN_WORD_ID,
        index_from=INDEX_FROM
    )

    # Pad word ID lists of dataset to uniform length.
    train_data = sequence.pad_sequences(train_data, maxlen=PAD_MAX_LENGTH, value=PAD_WORD_ID)
    test_data = sequence.pad_sequences(test_data, maxlen=PAD_MAX_LENGTH, value=PAD_WORD_ID)

    # Define model (neural network).
    model = Sequential()
    model.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim=64))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    # Build model.
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train model using custom Keras callback.
    monitor = KeruviRemoteMonitor(root=root_url, model_id='imdb')
    model.fit(
        train_data,
        train_labels,
        batch_size=512,
        epochs=20,
        validation_split=0.2,
        callbacks=[monitor],
        verbose=0
    )

    # Evaluate model.
    metrics = model.evaluate(test_data, test_labels, verbose=0)
    print(metrics)

    # Save fully trained model.
    model.save('imdb_model.h5')