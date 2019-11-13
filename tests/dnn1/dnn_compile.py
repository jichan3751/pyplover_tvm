## https://docs.tvm.ai/tutorials/relay_quick_start.html#sphx-glr-tutorials-relay-quick-start-py

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

### Load resnet ######

mod, params = relay.testing.resnet.get_workload(
    num_layers=18, batch_size=batch_size, image_shape=image_shape)

# set show_meta_data=True if you want to show meta data
# print(mod.astext(show_meta_data=False))


##### build ##############

opt_level = 3
target = tvm.target.cuda()
with relay.build_config(opt_level=opt_level):
    graph, lib, params = relay.build_module.build(
        mod, target, params=params)


###### Run local #########

# create random input
ctx = tvm.gpu()
# data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
data = np.random.uniform(size=data_shape).astype("float32")
# create module
module = graph_runtime.create(graph, lib, ctx)
print(module)

## method 1

# set input and parameters
module.set_input("data", data)
module.set_input(**params)
# run
module.run()

# get output
out = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()

# Print first 10 elements of output
print(out.flatten()[0:10])


######## save the graph, lib and params into separate files #############
from tvm.contrib import util

lib.export_library("deploy_lib.tar")

with open("deploy_graph.json", "w") as fo:
    fo.write(graph)
    # print(graph) # graph is just a json file

with open("deploy_param.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))
