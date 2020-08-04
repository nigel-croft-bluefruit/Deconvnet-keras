# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # VGG16
#
# Create VGG16 version of model for use with DeConvNet sample code
# This notebook is based on ESC_25_resnet18_02

# + [markdown] colab_type="text" id="BvFC409hZGYt"
# ## Download Data (takes a LONG time)

# + colab={"base_uri": "https://localhost:8080/", "height": 364} colab_type="code" id="uuguZBmZAYza" outputId="7d238ae2-e8b2-4b06-d173-6de12d7469f7"
# #!wget https://github.com/karoldvl/ESC-50/archive/master.zip

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="lkuO8l60HJBp" outputId="8b54e4e8-e63d-4e4b-c19a-ce26c90f8832"
# #!unzip master.zip

# + [markdown] colab_type="text" id="lHHQGjLNYxVN"
# ## Imports

# + colab={} colab_type="code" id="4EHj2A3YLAkh"
# %cd /home/jupyter/dev/tflow/deconvnet/Deconvnet-keras
import os
import datetime
from pathlib import Path
from os import path
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from tensorflow.keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt
import librosa.display
# %load_ext tensorboard

# + [markdown] colab_type="text" id="kOrHamV-45aG"
# ## Data prep

# + colab={} colab_type="code" id="X1PN-EBFd7Nr"
ROOT_DIR = Path("/home/jupyter/")
PATH_ESC50 = ROOT_DIR/"AudioCategoriserEquipment/ESC-50-master"
PATH_ESC50_AUDIO = PATH_ESC50/"audio"
PATH_ESC50_CSV  = PATH_ESC50/"meta/esc50.csv"
print(os.path.exists(PATH_ESC50))

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="c-S4FruxKQQX" outputId="fca3ee99-05c2-4335-b038-ff503c57f9ef"
esc50_df = pd.read_csv(PATH_ESC50_CSV)
print(esc50_df)
# -

IMAGE_CACHE = path.join(ROOT_DIR, "dev/tflow/cache/")
# !mkdir -p {IMAGE_CACHE}

esc25_categories =[
 'breathing',
 'car_horn',
 'cat',
 'chainsaw',
 'chirping_birds',
 'clapping',
 'coughing',
 'cow',
 'crickets',
 'dog',
 'door_wood_knock',
 'engine',
 'frog',
 'glass_breaking',
 'hen',
 'laughing',
 'rain',
 'rooster',
 'sheep',
 'siren',
 'sneezing',
 'snoring',
 'thunderstorm',
 'vacuum_cleaner',
  'wind']

esc25_df = esc50_df[esc50_df.category.isin(esc25_categories)]
print(esc25_df)

# +
from IPython.display import clear_output
import librosa

def extract_feature(filename):
    y, sr = librosa.load(filename, sr=44100)   
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S = librosa.power_to_db(S)
    S = np.transpose(S)
    S = np.expand_dims(S, 2)
    #S = tf.image.grayscale_to_rgb(S)
    return S

def get_feature_from_row(row):
    filename = PATH_ESC50_AUDIO/row['filename']
    category_label = row['category']
    data = extract_feature(filename)
    return (data, category_label)

def get_feature_set(metadata):
    number_of_rows = len(metadata)
    processed_rows = 0
    validation_set = []
    training_set = []
    for index, row in metadata.iterrows():
        processed_rows += 1  
        clear_output(wait=True)
        if row["fold"] == 1:  
            validation_set.append(get_feature_from_row(row))
        else:
            training_set.append(get_feature_from_row(row))
        progress = float(100 * processed_rows) / number_of_rows
        print(f"Current process: {progress:.2f}%")
    return validation_set, training_set


# -

validation_set, training_set = get_feature_set(esc25_df)

import random
random.shuffle(training_set)
len(training_set)

le = LabelEncoder()
train_x = np.array([data for data, label in training_set])
train_y = to_categorical(le.fit_transform(np.array([label for data, label in training_set])))
test_x = np.array([data for data, label in validation_set])
test_y = to_categorical(le.fit_transform(np.array([label for data, label in validation_set])))
print(test_y.shape)
print(train_y.shape)
print(test_x.shape)
print(train_x.shape)

# + [markdown] colab_type="text" id="ZKmrGuh2K5hX"
# #### Shaping the data

# + colab={} colab_type="code" id="6pmPzbq3NUh0"
timestamp = datetime.now().strftime("%d-%m-%y-%H-%M")
save_data = open(f"esc-25-a-{timestamp}.pickle", "wb")
pickle.dump((test_x, test_y, train_x, train_y), save_data)
save_data.close()

# + [markdown] colab_type="text" id="sOAsI_125AvG"
# ## Building the model

# + colab={} colab_type="code" id="QBPYncDM2aVs"
test_x, test_y, train_x, train_y = pickle.load(open("../../esc-25-a-11-06-20-10-28.pickle", "rb"))
# -

train_x.shape

# + colab={} colab_type="code" id="Zeif_bAXDWDD"
from tensorflow.keras.applications import vgg16

num_epochs = 40
num_batch_size = 32

num_labels = len(train_y[0])

input_layer = Input(shape=train_x.shape[1:])
#reshape_layer = Reshape((train_x.shape[1], train_x.shape[2], 3), input_shape = train_x.shape[1:])
reshape_layer = Conv2D(filters=3, kernel_size=1, input_shape=(train_x.shape[1:]))
base_model = vgg16.VGG16(input_shape=(431,128,3), weights='imagenet', include_top=False)
base_model.trainable = True

out = reshape_layer(input_layer)
out = base_model(out)
out = GlobalAveragePooling2D()(out)
out = Dropout(0.7)(out) #do we need this?
predictions = Dense(num_labels, activation='softmax')(out)

model = Model(inputs = input_layer, outputs = predictions)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="iK6yHpgme1iE" outputId="81c060b6-809d-4abf-c2ea-dfe42b018e1a"
#TODO - Add Training rate
learning_rate = ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=num_epochs,
    decay_rate=0.95)

optimizer = Adam(learning_rate=learning_rate)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

model.summary()
score = model.evaluate(test_x, test_y, verbose=0)
accuracy = 100*score[1]
print("Pre-training accuracy: %.4f%%" % accuracy)

# + [markdown] colab_type="text" id="y4EFlvxZA0ti"
# ## Training
# -

# Use generator instead of plain data array for training
import math
import random
class AugmentedData(Sequence):
    
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = np.copy(self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size])
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        
        batch_x = self._frequency_mask(batch_x)
        batch_x = self._roll(batch_x)
        return batch_x, batch_y
    
    def _roll(self, batch_x, max_shift_pct=0.7):
        direction = random.choice([-1, 1])
        width, height, c  = batch_x.shape[1:]
        roll_by = int(width*random.random()*max_shift_pct*direction)
        return tf.roll(batch_x, roll_by, axis=1)

    def _frequency_mask(self, batch_x, num_rows=8):
        channel_mean = np.mean(batch_x) #batch_x.contiguous().view(batch_x.size(0), -1).mean(-1).unsqueeze(-1).unsqueeze(-1)
        x, y, c  = batch_x.shape[1:]
        mask = tf.ones([x, num_rows,1]) * channel_mean
        mask = tf.expand_dims(mask, 0)
        start_row = random.randint(0, y-num_rows)
        batch_x[:,:,start_row:start_row+num_rows,:] = mask
        return batch_x


augmented_train = AugmentedData(train_x, train_y, num_batch_size)


def generate_image(spectro):
    plt.axis('off')
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(spectro)
    plt.plot()


a = tf.squeeze(augmented_train[0][0][0],2)
a.shape
generate_image(np.array(a))

# + colab={"base_uri": "https://localhost:8080/", "height": 940} colab_type="code" id="pkxCYEDWA2OU" outputId="e66efb3e-6b98-452b-fbe6-ceaeacb0ee6b"
# !rm -r logs/*
logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

start = datetime.now()

model.fit(augmented_train, epochs=num_epochs, \
          validation_data=(test_x, test_y), callbacks=[tensorboard_callback], verbose=1)

duration = datetime.now() - start

print("Training completed in time: ", duration)
# -

model.fit(augmented_train, epochs=20, \
          validation_data=(test_x, test_y), verbose=1)


model.fit(augmented_train, epochs=10, \
          validation_data=(test_x, test_y), callbacks=[tensorboard_callback], verbose=1)


timestamp = datetime.now().strftime("%d-%m-%y-%H-%M")
model.save(f"/home/jupyter/dev/tflow/saved-models/esc_25_vgg16-{timestamp}") 

# ### different LR

# + colab={} colab_type="code" id="Zeif_bAXDWDD"
from tensorflow.keras.applications import vgg16

num_epochs = 40
num_batch_size = 32

num_labels = len(train_y[0])

input_layer = Input(shape=train_x.shape[1:])
#reshape_layer = Reshape((train_x.shape[1], train_x.shape[2], 3), input_shape = train_x.shape[1:])
reshape_layer = Conv2D(filters=3, kernel_size=1, input_shape=(train_x.shape[1:]))
base_model = vgg16.VGG16(input_shape=(431,128,3), weights='imagenet', include_top=False)
base_model.trainable = True

out = reshape_layer(input_layer)
out = base_model(out)
out = GlobalAveragePooling2D()(out)
out = Dropout(0.7)(out) #do we need this?
predictions = Dense(num_labels, activation='softmax')(out)

model = Model(inputs = input_layer, outputs = predictions)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="iK6yHpgme1iE" outputId="81c060b6-809d-4abf-c2ea-dfe42b018e1a"
#TODO - Add Training rate
learning_rate = ExponentialDecay(
    initial_learning_rate=3e-5,
    decay_steps=num_epochs,
    decay_rate=0.95)

optimizer = Adam(learning_rate=learning_rate)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

model.summary()
score = model.evaluate(test_x, test_y, verbose=0)
accuracy = 100*score[1]
print("Pre-training accuracy: %.4f%%" % accuracy)
# -

model.fit(augmented_train, epochs=10, \
          validation_data=(test_x, test_y), verbose=1)

model.fit(augmented_train, epochs=20, \
          validation_data=(test_x, test_y), verbose=1)

model.fit(augmented_train, epochs=10, \
          validation_data=(test_x, test_y), verbose=1)

timestamp = datetime.now().strftime("%d-%m-%y-%H-%M")
model.save(f"/home/jupyter/dev/tflow/saved-models/esc_25_vgg16-{timestamp}") 

# + [markdown] colab_type="text" id="DLcqbgNK1_3c"
# ## Convert to TFLite

# + colab={} colab_type="code" id="Fp6UZf4j-VOk"
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# + colab={} colab_type="code" id="PEeyk0jvpBeR"
tflite_model = converter.convert()

# + colab={} colab_type="code" id="1o1zu890m4AF"
timestamp = datetime.now().strftime("%d-%m-%y-%H-%M")
with open(f"/home/jupyter/dev/tflow/saved-models/esc_25_vgg16-{timestamp}.tflite", "wb") as f:
    f.write(tflite_model)
# -

# # Testing script

import glob
from pathlib import Path
from os import path

recordings_dir = "/home/jupyter/AudioCategoriserEquipment-AudioSamples/"

audio_filepaths = []
for filepath in Path(recordings_dir).rglob('*.wav'):
    audio_filepaths.append(str(filepath))

filepaths_with_cat = [(filepath, path.dirname(filepath).split("/")[-1]) for filepath in audio_filepaths]

# +
import numpy as np
import librosa

try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    import tensorflow.lite as tflite

MODEL_PATH = "/home/jupyter/dev/tflow/saved-models/esc_25_vgg16-03-08-20-14-52.tflite"

def extract_feature(filename):
    y, sr = librosa.load(filename, sr=44100)   
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S = librosa.power_to_db(S)
    S = np.transpose(S)
    return S[np.newaxis, 0:431, :, np.newaxis]


def get_human_readable_output(output):
    results = [
        'breathing', 'car_horn', 'cat',
        'chainsaw', 'chirping_birds',
        'clapping', 'coughing', 'cow',
        'crickets', 'dog', 'door_wood_knock',
        'engine', 'frog', 'glass_breaking',
        'hen', 'laughing', 'rain', 'rooster',
        'sheep', 'siren', 'sneezing',
        'snoring', 'thunderstorm',
        'vacuum_cleaner', 'wind'
    ]

    output = list(output[0])

    largest = max(output)
    index = list(output).index(largest)
    return results[index]
    #result = f"{largest * 100:.2f}% confident that this is a {results[index]}"

    #formatted_output = [f"{o*100:05.2f}" for o in output]
    #stats = [f"{x}%: {results[i]}" for i, x in enumerate(formatted_output)]

    #return result, "\n".join(stats)


def infer_from_file(filename, model_path=MODEL_PATH):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = extract_feature(filename)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predicted_data_vector = interpreter.get_tensor(output_details[0]['index'])

    return get_human_readable_output(predicted_data_vector)


# +
import time

start = time.time()
results = []
for filepath, cat in filepaths_with_cat:
    results.append((infer_from_file(filepath), cat))
end = time.time()

# +
correct = 0
for result, expected in results:
    if result == expected: correct +=1 
        
print(f"Got {correct} out of {len(results)}")
print(f"{correct/len(results)*100:.2f}% correct")
print(f"Took {end - start}")
# -

# esc_25_simple_05:
#
#     Got 88 out of 192
#     45.83% correct
#     Took 88.54123330116272
#     
# esc_25_resnet18-02-24-07-20-13-26:
#
#     Got 88 out of 192
#     45.83% correct
#     Took 50.27479147911072


