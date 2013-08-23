import numpy as np
import pyfits as pf
from scipy import signal
from scipy import ndimage
import scipy
import time

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import pycuda.gpuarray as cua
    from pycuda.compiler import compile
except ImportError, e:
    print 'PyCUDA required, please install'
    print e
    import sys
    sys.exit(-9)

try:
    import pyfft.cuda as cufft
except ImportError, e:
    print 'PyFFT required, please install'
    print e
    import sys
    sys.exit(-9)


cubin = compile(open('/Users/sammy/EUCLID/vissim-python/sandbox/gputools.cu').read(), keep=True)


def example(mode='same'):
    print 'Convlution mode =', mode

    memory = pycuda.driver.mem_get_info()
    print 'Free Memory = %.1f MB and Total = %.1f MB\n\n' % (memory[0] / 1.049e+6, memory[1] / 1.049e+6)

    #random data
    np.random.seed(1)

    #data
    image = pf.getdata('/Users/sammy/EUCLID/vissim-python/objects/galaxy37.fits')
    image = ndimage.zoom(image, 4.0, order=0)
    image[image <= 0] = 1e-8
    image /= np.max(image)
    image *= 4000.
    image = image.astype(np.float32)
    scipy.misc.imsave('original.jpg', np.log10(image))

    #kernel
    kernel = pf.getdata('/Users/sammy/EUCLID/vissim-python/data/psf4x.fits')
    kernel /= np.sum(kernel)
    kernel = kernel.astype(np.float32)
    scipy.misc.imsave('kernel.jpg', np.log10(kernel))

    #FFT padding size (next power two)
    sw = np.array(image.shape)
    sf = np.array(kernel.shape)

    #nominal
    sfft = sw + sf - 1

    #save memory
    #sfft = [max(sw[0], sf[0]) + 1, max(sw[1], sf[1]) + 1]  #to save memory

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
    fft_gpu = zeropadToGPU(image, sfft_gpu, dtype='complex')

    # Create FFT plan and compute FFT of the image
    plan = cufft.Plan(fft_gpu.shape, fast_math=True)
    plan.execute(fft_gpu)

    #pad the kernel with zeros and copy to the GPU
    u_gpu = zeropadToGPU(kernel, sfft_gpu, dtype='complex')

    # Compute FFT of the kernel
    plan.execute(u_gpu)

    #do multiplication in the Fourier space
    try:
        tmp = fft_gpu * u_gpu
    except:
        memory = pycuda.driver.mem_get_info()
        print 'Free Memory = %.1f MB' % (memory[0] / 1.049e+6)
        print 'No memory for multiplication on the GPU, doing multiplication on CPU...'
        tmp = fft_gpu.get() * u_gpu.get()
        fft_gpu.gpudata.free()
        u_gpu.gpudata.free()
        #copy back to GPU for inverse transformation
        tmp = cua.to_gpu(tmp)

    # and compute inverse Fourier transform
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

    del tmp

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

    r = result.copy()
    neg = r < 0.
    r[neg] *= -1.
    scipy.misc.imsave('cuda%s.jpg' % mode, np.log10(r))
    scipy.misc.imsave('scipy%s.jpg' % mode, np.log10(conv))

    print '\n\nDifference:'
    if 'full' in mode:
        rtol = 1.
        print 'rtol = %e' % rtol
        print np.testing.assert_allclose(result[100:-100, 100:-100], conv[100:-100, 100:-100], rtol=rtol,
                                         atol=0.0, verbose=True)
    else:
        rtol = 1e-4
        print 'rtol = %e' % rtol
        print np.testing.assert_allclose(result, conv, rtol=rtol, atol=0.0, verbose=True)


def cropGPU(gpuArray, size, offset=(0, 0), block_size=(32, 32, 1)):
    """
    Crop an image array that is on the GPU to a new size.

    :param gpuArray: image array to be cropped
    :type gpuArray: GPU array
    :param size: size to which the array is crooped to (y, x)
    :type size: tuple
    :param offset: apply offset?
    :type offset: tuple
    :param block_size: CUDA block_size
    :param block_size: tuple

    :return: cropped array
    :rtype: GPU array
    """

    sfft = gpuArray.shape

    grid_size = (int(np.ceil(float(sfft[1])/block_size[1])),
                 int(np.ceil(float(sfft[0])/block_size[0])))

    if gpuArray.dtype == np.float32:
        mod = cuda.module_from_buffer(cubin)
        cropKernel = mod.get_function("crop_Kernel")

    elif gpuArray.dtype == np.complex64:
        mod = cuda.module_from_buffer(cubin)
        cropKernel = mod.get_function("crop_ComplexKernel")
    else:
        print 'Incorrect data type in cropGPU'
        return None

    x_cropped_gpu = cua.zeros(tuple((int(size[0]),int(size[1]))), np.float32)

    cropKernel(x_cropped_gpu.gpudata, np.int32(size[0]), np.int32(size[1]),
               gpuArray.gpudata, np.int32(sfft[0]), np.int32(sfft[1]),
               np.int32(offset[0]), np.int32(offset[1]),
               block=block_size, grid=grid_size)

    return x_cropped_gpu


def zeropadToGPU(array, size, offset=(0, 0), dtype='real', block_size=(32, 32, 1)):
    """
    Zero pad the input array and transfer it to the GPU memory if not there yet

    :param array: input array to be zeropadded and transferred
    :type array: ndarray
    :param size: size of the array (y, x)
    :type size: tuple
    :param offset: apply offset?
    :type offset: tuple
    :param dtype: data type, either real or complex
    :type: str
    :param block_size: CUDA block_size
    :param block_size: tuple

    :return: zero padded array that resides in the GPU memory
    :rtype: GPUarray
    """
    grid_size = (int(np.ceil(float(size[1])/block_size[1])),
                 int(np.ceil(float(size[0])/block_size[0])))

    ay, ax = array.shape
    ay = np.int32(ay)
    ax = np.int32(ax)

    offsetx = np.int32(offset[0])
    offsety = np.int32(offset[1])

    sy = np.int32(size[0])
    sx = np.int32(size[1])

    if array.__class__ == np.ndarray:
        array_gpu = cua.to_gpu(array)
        #array_gpu = cua.to_gpu_async(array.astype(np.float32))
    elif array.__class__ == cua.GPUArray:
        array_gpu = array
    else:
        print 'ERROR: Array type neither NumPy or GPUArray'
        return None

    if dtype == 'real':
        mod = cuda.module_from_buffer(cubin)
        zeroPadKernel = mod.get_function("zeroPadKernel")

        output = cua.zeros(size, np.float32)

        zeroPadKernel(output.gpudata, sy, sx, array_gpu.gpudata, ay, ax,
                      offsetx, offsety, block=block_size, grid=grid_size)
    elif dtype == 'complex':
        mod = cuda.module_from_buffer(cubin)
        zeroPadComplexKernel = mod.get_function("zeroPadComplexKernel")

        output = cua.zeros(size, np.complex64)

        zeroPadComplexKernel(output.gpudata, sy, sx, array_gpu.gpudata, ay, ax,
                             offsetx, offsety, block=block_size, grid=grid_size)
    else:
        print 'Incorrect data type in zeropadToGPU'
        return None

    return output


if __name__ == "__main__":
    example()
    example(mode='full')
