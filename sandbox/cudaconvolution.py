import numpy as np
import pyfits as pf
from scipy import signal
from scipy import ndimage
import scipy
import time

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as cua
from pycuda.compiler import compile

import pyfft.cuda as cufft


cubin = compile(open('/Users/sammy/EUCLID/vissim-python/sandbox/gputools.cu').read(), keep=True)


def example(mode='same'):
    #random data
    np.random.seed(1)

    #data
    image = pf.getdata('/Users/sammy/EUCLID/vissim-python/objects/galaxy37.fits')
    image = ndimage.zoom(image, 12.0, order=0)
    image[image <= 0] = 1e-8
    image /= np.max(image)
    image *= 4000.
    image = image.astype(np.float32)
    scipy.misc.imsave('original.jpg', np.log10(image))

    #kernel
    kernel = pf.getdata('/Users/sammy/EUCLID/vissim-python/data/psf12x.fits')
    kernel /= np.sum(kernel)
    kernel = kernel.astype(np.float32)
    scipy.misc.imsave('kernel.jpg', np.log10(kernel))

    #FFT padding size (next power two)
    sw = np.array(image.shape)
    sf = np.array(kernel.shape)
    #sfft = sw + sf - 1     #to save memory...
    sfft = [max(sw[0], sf[0]) + 1, max(sw[1], sf[1]) + 1]
    sfft_gpu = (2 ** np.ceil(np.log2(sfft)))
    sfft_gpu = (int(sfft_gpu[0]), int(sfft_gpu[1]))

    print 'Image size:', image.shape
    print 'Kernel size:', kernel.shape
    print 'Zero padded FFT size:', sfft_gpu

    #output dimension
    sx = np.array(image.shape)
    if mode == 'valid':
        sy = sx - sf + 1
    elif mode == 'same':
        sy = sx
    elif mode == 'full':
        sy = sx + sf - 1
    else:
        sy = sx

    print "--------------------------"
    print "Starting CUDA calculations"
    start = time.clock()

    #pad the image with zeros and copy to the GPU
    fft_gpu = ZeropadToGPU(image, sfft_gpu, dtype='complex')
    #pad the kernel with zeros and copy to the GPU
    u_gpu = ZeropadToGPU(kernel, sfft_gpu, dtype='complex')

    # Create FFT plan and compute FFT of the image
    plan = cufft.Plan(fft_gpu.shape)
    plan.execute(fft_gpu)

    # Compute FFT of the kernel, do multiplication in Fourier space
    # and compute inverse Fourier transform
    plan.execute(u_gpu)
    tmp = fft_gpu * u_gpu
    plan.execute(tmp, inverse=True)

    #crop the result to right size
    if mode == 'valid':
        result = cropGPU(tmp, sy, sf - 1)
    elif mode == 'same':
        result = cropGPU(tmp, sy, np.floor(sf / 2. - 1))
    elif mode == 'full':
        result = cropGPU(tmp, sy)

    result = result.get().real

    cudatime = time.clock()-start
    print "Time elapsed: %.4f" % cudatime
    print "--------------------------"

    #scipy
    print "SciPy FFT convolution on CPU"
    start = time.clock()
    conv = signal.fftconvolve(image, kernel, mode=mode)
    sptime = time.clock()-start
    print "Time elapsed: %.4f" % sptime
    print "--------------------------"

    print 'CUDA is a factor of %.2f faster' % (sptime / cudatime)
    print "--------------------------\n\n\n\n\n\n\n"

    print np.max(result), np.max(conv)

    scipy.misc.imsave('kernel.jpg', np.log10(kernel))
    scipy.misc.imsave('cuda.jpg', np.log10(result))
    scipy.misc.imsave('scipy.jpg', np.log10(conv))

    print '\n\nDifference:'
    if 'full' or 'same' in mode:
        print '> 1e-4?'
        print np.testing.assert_allclose(result[100:-100, 100:-100], conv[100:-100, 100:-100], rtol=1e-4)
    else:
        print '> 1e-5?'
        print np.testing.assert_allclose(result, conv, rtol=1e-5)


def cropGPU(x_gpu, sz, offset=(0,0)):

    sfft = x_gpu.shape

    block_size = (32, 32, 1)

    grid_size = (int(np.ceil(float(sfft[1])/block_size[1])),
                 int(np.ceil(float(sfft[0])/block_size[0])))

    if x_gpu.dtype == np.float32:
        mod = cuda.module_from_buffer(cubin)
        cropKernel = mod.get_function("crop_Kernel")

    elif x_gpu.dtype == np.complex64:
        mod = cuda.module_from_buffer(cubin)
        cropKernel = mod.get_function("crop_ComplexKernel")

    x_cropped_gpu = cua.empty(tuple((int(sz[0]),int(sz[1]))), np.float32)

    cropKernel(x_cropped_gpu.gpudata, np.int32(sz[0]), np.int32(sz[1]),
               x_gpu.gpudata, np.int32(sfft[0]), np.int32(sfft[1]),
               np.int32(offset[0]), np.int32(offset[1]),
               block=block_size, grid=grid_size)

    return x_cropped_gpu


def ZeropadToGPU(x, sz, offset=(0,0), dtype='real', block_size=(32, 32, 1)):


    grid_size = (int(np.ceil(float(sz[1])/block_size[1])),
                 int(np.ceil(float(sz[0])/block_size[0])))

    sx = x.shape

    if x.__class__ == np.ndarray:
        x  = np.array(x).astype(np.float32)
        x_gpu = cua.to_gpu(x)
    elif x.__class__ == cua.GPUArray:
        x_gpu = x

    if dtype == 'real':

        mod = cuda.module_from_buffer(cubin)
        zeroPadKernel = mod.get_function("zeroPadKernel")

        x_padded_gpu = cua.zeros(tuple((int(sz[0]),int(sz[1]))), np.float32)

        zeroPadKernel(x_padded_gpu.gpudata, np.int32(sz[0]), np.int32(sz[1]),
                      x_gpu.gpudata, np.int32(sx[0]), np.int32(sx[1]), np.int32(offset[0]), np.int32(offset[1]),
                      block=block_size, grid=grid_size)
    elif dtype == 'complex':

        mod = cuda.module_from_buffer(cubin)
        zeroPadComplexKernel = mod.get_function("zeroPadComplexKernel")

        x_padded_gpu = cua.zeros(tuple((int(sz[0]),int(sz[1]))), np.complex64)

        zeroPadComplexKernel(x_padded_gpu.gpudata, np.int32(sz[0]), np.int32(sz[1]),
                             x_gpu.gpudata, np.int32(sx[0]), np.int32(sx[1]), np.int32(offset[0]), np.int32(offset[1]),
                             block=block_size, grid=grid_size)

    return x_padded_gpu


if __name__ == "__main__":
    example()