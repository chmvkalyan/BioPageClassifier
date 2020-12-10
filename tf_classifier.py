import os
import collections
from tensorflow.keras.utils import plot_model
from utils import *
import numpy as np

FilenameCollection = collections.namedtuple('FilenameCollection', 'texts_path urls_path')

# Seed for numpy randomizer
SEED = 1337

# Number of epochs for training
EPOCH = 5

# Batch size
BATCH_SIZE = 512

# Embedding
EMBEDDING = "https://tfhub.dev/google/nnlm-en-dim50/2"

# Tensor parameters
# ---------------------------------------------------------
# The name of the saved dataframe (helps with caching)
KERAS_TRAINING_SET_NAME = "training-data/keras_training_set.plk"

# The trained model name
TENSOR_MODEL_NAME = "bio-model"

# The filename in which the graphical picture of Keras NN will be saved
TENSOR_MODEL_ARCH_PLOT_NAME = "{}-arch-plot.png".format(TENSOR_MODEL_NAME)

# The filename in which the training history plot will be saved
TENSOR_MODEL_HISTORY_PLOT_NAME = "{}-history-plot.png".format(TENSOR_MODEL_NAME)

# Training data parameters
# ---------------------------------------------------------
# Positive training data locations
POSITIVE_TRAINING_DATA_FILENAMES = FilenameCollection(
    texts_path="training-data/bios.txt", urls_path="training-data/urls.text")

# Negative training data locations
NEGATIVE_TRAINING_DATA_FILENAMES = FilenameCollection(
    texts_path="training-data/neg-texts.txt", urls_path="training-data/neg-urls.txt")

np.random.seed(SEED)

# Load compressed models from tensorflow_hub
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"


def create_model():
    hub_layer = hub.KerasLayer(EMBEDDING, input_shape=[], dtype=tf.string, trainable=True)
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.summary()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['acc'])
    return model


def create_model_complex():
    # Uses Keras Sequential Model
    model = tf.keras.Sequential()

    # Hub Layer = Pretrained Model
    model.add(hub.KerasLayer(EMBEDDING, input_shape=[], dtype=tf.string, trainable=True))

    # Fine tune model
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # Compile with Binary Cross Entropy
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['acc'])
    return model


def create_lstm_model(encoder):
    # Uses Keras Sequential Model
    model = tf.keras.Sequential()
    if encoder:
        model.add(encoder)

    model.add(tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True
    ))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))),
    model.add(tf.keras.layers.Dense(64, activation='relu')),
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # Compile with Binary Cross Entropy
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['acc'])
    return model


def evaluate_model(model_name, model, test_dataset):
    results = model.evaluate(test_dataset.batch(BATCH_SIZE), verbose=2)
    for name, value in zip(model.metrics_names, results):
        print("Model name: %s - %s: %.3f" % (model_name, name, value))


def run_tensor_trainer():
    # Create training dataset
    full_dataset, train_dataset, val_dataset, test_dataset = create_tensor_dataset(
        KERAS_TRAINING_SET_NAME,
        POSITIVE_TRAINING_DATA_FILENAMES.texts_path,
        NEGATIVE_TRAINING_DATA_FILENAMES.texts_path)

    # Encoding
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=800)
    encoder.adapt(full_dataset.map(lambda text,label: text))

    # Create model and train
    model = create_lstm_model(encoder=encoder)
    plot_model(model, to_file=TENSOR_MODEL_ARCH_PLOT_NAME, show_shapes=True, show_layer_names=True)

    # cp = tf.keras.callbacks.ModelCheckpoint(filepath='best_weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # Early stopping
    # es = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto',baseline=None, restore_best_weights=False)

    history = model.fit(
        train_dataset.shuffle(10000).batch(BATCH_SIZE),
        epochs=EPOCH, validation_data=val_dataset.batch(BATCH_SIZE), verbose=1) # callbacks=[cp, es]

    # model.load_weights('best_weights.hdf5')
    evaluate_model("orig", model, test_dataset)
    plot_history(history, plot_filename=TENSOR_MODEL_HISTORY_PLOT_NAME)

    # Saving the model
    model.save(TENSOR_MODEL_NAME)
    model = tf.keras.models.load_model(TENSOR_MODEL_NAME)

    evaluate_model("loaded", model, test_dataset)


# Entry Point
run_tensor_trainer()