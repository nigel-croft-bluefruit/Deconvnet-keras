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
#
# ## Look at dog.wav in block2_conv1 & block5_conv3
# This clip is categorised as 97.8% sneeze - can we find out why?

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

model.layers[2].summary()


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
        #print(f"{i}")
        deconv_layers[i].up(deconv_layers[i - 1].up_data)

    return deconv_layers[-1].up_data


def backward_pass(deconv_layers, output, feature_to_visualize, visualize_mode):
    assert output.ndim == 2 or output.ndim == 4
    if output.ndim == 2:
        feature_map = output[:, feature_to_visualize]
    else:
        feature_map = output[:, :, :, feature_to_visualize]
    if 'max' == visualize_mode:
        if not isinstance(feature_map, np.ndarray):
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


# +
def max_activation(output):
    a = np.mean(np.mean(output[0,:,:,:],axis=0), axis=0)
    return (-a).argsort()

def multiplot(layers, mode='all', rows=None, cols=2):
    if rows == None:
        rows = len(layers) // cols
        
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

recordings_dir = "/home/jupyter/dev/tflow/unit_test_files"

# ### Look at block2_conv1 (near input)

LAYER_NAME = 'block2_conv1'
deconv_layers = create_deconv(model, LAYER_NAME)

deconv_layers

s = extract_feature(path.join(recordings_dir,"coughing.wav"))
show_image(s[0,:,:,0],'magma')

output = forward_pass(deconv_layers, extract_feature(path.join(recordings_dir,"coughing.wav")))

result = backward_pass(deconv_layers, output, 46, 'all')
show_image(result)

# There is a lot of detail being found in the "silence" - lots of edges! Could we threshold the clips so that everything below a certain threshold gets set to the same value?

result = backward_pass(deconv_layers, output, 46, 'max')
show_image(result)

max_act = max_activation(output)
multiplot(max_act[:32], cols=4)


# At this level we are just finding edges mostly?

multiplot(max_act[:32],'max', cols=4)

# ### Look at block5_pool (last convolution before output)

LAYER_NAME = 'block5_pool'
deconv_layers = create_deconv(model, LAYER_NAME)

output = forward_pass(deconv_layers, extract_feature(path.join(recordings_dir,"coughing.wav")))

multiplot(range(32), cols=4)

# A lot of filters are not activating at all at the final layer

max_act = max_activation(output)
multiplot(max_act[:32], cols=4)

multiplot(max_act[:32],'max', cols=4)

# At this layer, the 'max' and 'all' look pretty similar

#Show most activated filter full size
result = backward_pass(deconv_layers, output, 347, 'all')
show_image(result)

show_image(result,'magma')


