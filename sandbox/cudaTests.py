"""
Some tests with CUDA
"""
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray


def firstExample():
    #generate random data and convert to 32bit
    a = np.random.randn(4,4)
    a = a.astype(np.float32)  #note that results are incorrect if this is not done!

    #allocate memory
    a_gpu = cuda.mem_alloc(a.nbytes)

    #transfer to the GPU memory
    cuda.memcpy_htod(a_gpu, a)

    #source module C-code
    mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
  """)

    print 'With SourceModule:'
    func = mod.get_function("doublify")
    func(a_gpu, block=(4,4,1))

    a_doubled = np.empty_like(a)
    cuda.memcpy_dtoh(a_doubled, a_gpu)
    print a_doubled
    print a*2

    print
    print 'GPU array (64bit):'
    a_gpu = gpuarray.to_gpu(np.random.randn(4,4))
    a_doubled = (2*a_gpu).get()
    print a_doubled
    print a_gpu*2


if __name__ == "__main__":
    firstExample()