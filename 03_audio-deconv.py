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

# # Deconvolution for layer visualisation
#
# Investigate deconvnets. Initially, we are copying an example that uses a VGG rather than a resnet. It is not currently clear what problems are associated with resnets, or if they can even be used for this.
#
# This notebook reproduces the results of https://github.com/jalused/Deconvnet-keras/blob/master/Deconvnet-keras.py after converting to use TF2.2

# %cd /home/jupyter/dev/tflow/deconvnet/Deconvnet-keras
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import os
import sys
from pathlib import Path
from os import path

from Deconvnet import *
import matplotlib.pyplot as plt
import librosa

# -


model = tf.keras.models.load_model("/home/jupyter/dev/tflow/saved-models/esc_25_vgg16-03-08-20-14-49")


model.summary()

LAYER_NAME = 'block4_conv2'
deconv_layers = create_deconv(model, LAYER_NAME)

deconv_layers


def extract_feature(filename):
    y, sr = librosa.load(filename, sr=44100)   
    #S = librosa.feature.melspectrogram(y=y, sr=sr)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        fmax=(sr / 2.0),
        fmin=0.0,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    S = librosa.power_to_db(S)
    S = np.transpose(S)
    return S[np.newaxis, 0:431, :, np.newaxis]



def main(filepath, feature_to_visualize, visualize_mode='all'):

    # Load data and preprocess
    #img = Image.open(image_path)
    #img = img.resize((224, 224))
    #img_array = np.array(img)
    #img_array = img_array[np.newaxis, :]
    #img_array = img_array.astype(np.float)
    #img_array = imagenet_utils.preprocess_input(img_array)
    
    data = extract_feature(filepath)
    deconv = visualize(deconv_layers, data, 
            feature_to_visualize, visualize_mode)
    
    return deconv


recordings_dir = "/home/jupyter/dev/tflow/unit_test_files"

result = main(path.join(recordings_dir,"vacuum_cleaner.wav"), 46)

import librosa.display
def generate_image(spectro):
    spectro = np.rot90(spectro)
    spectro = spectro[::-1,:]
    plt.axis('off')
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(spectro)
    plt.plot()


def show_image(spectro, cmap='gray'):
    _,ax = plt.subplots(figsize=(10,10))

    spectro = np.rot90(spectro)
    ax.axis('off')
    #ax.set_title(f"{layers[i]}:")
    ax.imshow(spectro, cmap=cmap)
    #ax.imshow(spectro)



def forward_pass(deconv_layers, data):
    # Forward pass
    deconv_layers[0].up(data)
    for i in range(1, len(deconv_layers)):
        print(f"{i}")
        #print(f"{i}: {deconv_layers[i].name}")
        deconv_layers[i].up(deconv_layers[i - 1].up_data)

    return deconv_layers[-1].up_data



output = forward_pass(deconv_layers, extract_feature(path.join(recordings_dir,"vacuum_cleaner.wav")))


def backward_pass(deconv_layers, output, feature_to_visualize, visualize_mode):
    assert output.ndim == 2 or output.ndim == 4
    if output.ndim == 2:
        feature_map = output[:, feature_to_visualize]
    else:
        feature_map = output[:, :, :, feature_to_visualize]
    if 'max' == visualize_mode:
        feature_map = feature_map.numpy()
        max_activation = feature_map.max()
        temp = feature_map == max_activation
        feature_map = feature_map * temp
    elif 'all' != visualize_mode:
        print('Illegal visualize mode')
        sys.exit()
    output = np.zeros_like(output)
    if 2 == output.ndim:
        output[:, feature_to_visualize] = feature_map
    else:
        output[:, :, :, feature_to_visualize] = feature_map

    # Backward pass
    deconv_layers[-1].down(output)
    for i in range(len(deconv_layers) - 2, -1, -1):
        #print(f"d: {i}")
        deconv_layers[i].down(deconv_layers[i + 1].down_data)
    deconv = deconv_layers[0].down_data
    deconv = deconv.squeeze()
    
    return deconv



s = extract_feature(path.join(recordings_dir,"vacuum_cleaner.wav"))
show_image(s[0,:,:,0],'magma')

result = backward_pass(deconv_layers, output, 46, 'all')
show_image(result)

result = backward_pass(deconv_layers, output, 46, 'max')
show_image(result)


# +
def max_activation(output):
    a = np.mean(np.mean(output[0,:,:,:],axis=0), axis=0)
    return (-a).argsort()

def multiplot(layers, mode='all', rows=4, cols=2):
    _,axes = plt.subplots(rows,cols,figsize=(15,10))

    for i, ax in enumerate(axes.flatten()):
        if i >= len(layers):
            break
        r = backward_pass(deconv_layers, output, layers[i], mode)
        spectro = np.rot90(r)
        #spectro = spectro[::-1,:]
        ax.axis('off')
        ax.set_title(f"{layers[i]}:")
        ax.imshow(spectro, cmap='gray')



# -

max_act = max_activation(output)
multiplot(max_act[:8])


multiplot(max_act[:8],'max')


