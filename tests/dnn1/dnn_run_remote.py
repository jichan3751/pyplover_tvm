
import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import rpc
from tvm.contrib import graph_runtime


batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

np.random.seed(0)


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

data = np.random.uniform(size=data_shape).astype("float32")

module = graph_runtime.create(loaded_graph, remote_lib, ctx)

# method 1
input_data = tvm.nd.array(data)
module.load_params(loaded_params)
module.run(data=input_data)

out_deploy = module.get_output(0).asnumpy()

# Print first 10 elements of output
print(out_deploy.flatten()[0:10])
