
import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime


batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

np.random.seed(0)

ctx = tvm.gpu()

# load the module back.
loaded_graph = open("deploy_graph.json").read()
loaded_lib = tvm.module.load("deploy_lib.tar")
loaded_params = bytearray(open("deploy_param.params", "rb").read())

data = np.random.uniform(size=data_shape).astype("float32")

module = graph_runtime.create(loaded_graph, loaded_lib, ctx)


# method 1
input_data = tvm.nd.array(data)
module.load_params(loaded_params)
module.run(data=input_data)

out_deploy = module.get_output(0).asnumpy()

# Print first 10 elements of output
print(out_deploy.flatten()[0:10])
