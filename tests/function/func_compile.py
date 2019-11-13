
import numpy as np

import tvm
from tvm import rpc
from tvm.contrib import util


## test cross compile


n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")

print(type(C))

s = tvm.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=64)

# bind

# looks like this kind of things are also used in cuda?
s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
s[C].bind(tx, tvm.thread_axis("threadIdx.x"))


# compile target
# target = tvm.target.cuda()
tgt = 'cuda'
fadd = tvm.build(s, [A, B, C], tgt, name="myadd")

# save the lib at a local temp folder

path = './lib.tar'
fadd.export_library(path)