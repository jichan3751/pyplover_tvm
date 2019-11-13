## https://docs.tvm.ai/tutorials/relay_quick_start.html#sphx-glr-tutorials-relay-quick-start-py

import os
import numpy as np
from PIL import Image
import keras
from keras.applications.mobilenet_v2 import MobileNetV2
import tvm
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import util, ndk, graph_runtime as runtime
from tvm.contrib.download import download_testdata

np.random.seed(0)


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


##### synset is used to transform the label from number of ImageNet class to the word human can understand.####
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'imagenet1000_clsid_to_human.txt'
synset_path = download_testdata(synset_url, synset_name, module='data')
with open(synset_path) as f:
    synset = eval(f.read())


##### remote run #####


# local_demo = True
local_demo = False

if local_demo:
    remote = rpc.LocalSession()

else:
    print('remote')
    # The following is my environment, change this to the IP address of your target device
    host = '127.0.0.1'
    port = 9090
    remote = rpc.connect(host, port)

path = "deploy_lib.tar"
remote.upload(path)
remote_lib = remote.load_module(path)

ctx = remote.gpu()

# load the module back.
loaded_graph = open("deploy_graph.json").read()
loaded_params = bytearray(open("deploy_param.params", "rb").read())

module = runtime.create(loaded_graph, remote_lib, ctx)

# set parameter (upload params to the remote device. This may take a while)
input_name = 'input_1'
input_data = tvm.nd.array(x.astype(dtype))

# module.set_input(**loaded_params)
# module.set_input(input_name, tvm.nd.array(x.astype(dtype)))
# module.run()

module.load_params(loaded_params)
module.set_input(input_name, tvm.nd.array(x.astype(dtype))) # key = input_name, value = array
module.run()

# get output
out = module.get_output(0)

# get top1 result
top1 = np.argmax(out.asnumpy())
print('TVM prediction top-1: {}'.format(synset[top1]))

print('Evaluate inference time cost...')
ftimer = module.module.time_evaluator('run', ctx, number=1, repeat=10)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print('Mean inference time (std dev): %.2f ms (%.2f ms)' % (np.mean(prof_res),
                                                            np.std(prof_res)))

