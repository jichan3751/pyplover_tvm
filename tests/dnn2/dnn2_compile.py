## https://docs.tvm.ai/tutorials/relay_quick_start.html#sphx-glr-tutorials-relay-quick-start-py

import os
import numpy as np
from PIL import Image # pip install pillow
import keras
from keras.applications.mobilenet_v2 import MobileNetV2
import tvm
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import util, ndk, graph_runtime as runtime
from tvm.contrib.download import download_testdata

np.random.seed(0)

### load pretrained model ########

keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
weights_url = ''.join(['https://github.com/JonathanCMitchell/',
                       'mobilenet_v2_keras/releases/download/v1.1/',
                       'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5'])
weights_file = 'mobilenet_v2_weights.h5'
weights_path = download_testdata(weights_url, weights_file, module='keras')
keras_mobilenet_v2 = MobileNetV2(alpha=0.5, include_top=True, weights=None,
                                input_shape=(224, 224, 3), classes=1000)
keras_mobilenet_v2.load_weights(weights_path)


######## load test set ###########
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_name = 'cat.png'
img_path = download_testdata(img_url, img_name, module='data')
image = Image.open(img_path).resize((224, 224))
dtype = 'float32'

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

x = transform_image(image)




local_demo = True
target = tvm.target.cuda()

input_name = 'input_1'
shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_keras(keras_mobilenet_v2, shape_dict)

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod, target=target,params=params)

# After `relay.build`, you will get three return values: graph,
# library and the new parameter, since we do some optimization that will
# change the parameters but keep the result of model as the same.

# Save the library at local temporary directory.

lib.export_library("deploy_lib.tar")

with open("deploy_graph.json", "w") as fo:
    fo.write(graph)
    # print(graph) # graph is just a json file

with open("deploy_param.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))

