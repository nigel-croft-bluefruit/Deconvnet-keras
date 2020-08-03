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

import os
import sys
sys.path.append(os.getcwd())
from Deconvnet import *
import matplotlib.pyplot as plt


model = vgg16.VGG16(weights = 'imagenet', include_top = True)


def main(image_path, layer_name, feature_to_visualize, visualize_mode='all'):

    #model = vgg16.VGG16(weights = 'imagenet', include_top = True)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    if not layer_name in layer_dict:
        print('Wrong layer name')
        sys.exit()

    # Load data and preprocess
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    #img_array = np.transpose(img_array, (2, 0, 1))
    img_array = img_array[np.newaxis, :]
    img_array = img_array.astype(np.float)
    img_array = imagenet_utils.preprocess_input(img_array)
    
    deconv = visualize(model, img_array, 
            layer_name, feature_to_visualize, visualize_mode)
    
    return deconv


# #%debug
result = main('./husky.jpg', 'block4_conv2', 46)

    # postprocess and save image
    #deconv = np.transpose(deconv, (1, 2, 0))
    deconv = result
    deconv = deconv - deconv.min()
    deconv *= 1.0 / (deconv.max() + 1e-8)
    deconv = deconv[:, :, ::-1]
    uint8_deconv = (deconv * 255).astype(np.uint8)
    img = Image.fromarray(uint8_deconv, 'RGB')
    #img.save('results/{}_{}_{}.png'.format(layer_name, feature_to_visualize, visualize_mode))
    plt.axis('off')
    #plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    plt.imshow(img)


def show_result(deconv):
    deconv = deconv - deconv.min()
    deconv *= 1.0 / (deconv.max() + 1e-8)
    deconv = deconv[:, :, ::-1]
    uint8_deconv = (deconv * 255).astype(np.uint8)
    img = Image.fromarray(uint8_deconv, 'RGB')
    #img.save('results/{}_{}_{}.png'.format(layer_name, feature_to_visualize, visualize_mode))
    plt.axis('off')
    #plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    plt.imshow(img)


result = main('./husky.jpg', 'block4_conv2', 46, visualize_mode='max')

show_result(result)

model.summary()


