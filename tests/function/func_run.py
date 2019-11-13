## test cross compile
# https://docs.tvm.ai/tutorials/cross_compilation_and_rpc.html#sphx-glr-tutorials-cross-compilation-and-rpc-py

import numpy as np

import tvm
from tvm import rpc
from tvm.contrib import util

n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")

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

path = './lib.tar'

remote.upload(path)
fadd = remote.load_module('lib.tar')

# available things https://docs.tvm.ai/api/python/rpc.html#tvm.rpc.Server
ctx = remote.gpu()

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

print(c.asnumpy())
