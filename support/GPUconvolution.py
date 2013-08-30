"""
This file contains functions to run convolutions on an NVIDIA GPU using CUDA.

:requires: PyCUDA
:requires: PyFFT
:requires: NumPy

:version: 0.2

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk
"""
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as cua
from pycuda.compiler import compile
import pyfft.cuda as cufft
import numpy as np

#currently hardcoded...
cubin = compile(open('/Users/sammy/EUCLID/vissim-python/support/gputools.cu').read(), keep=True)


def convolve(image, kernel, mode='same', saveMemory=False):
    """
    Convolves the input image with a given kernel.
    Current forces the image and kernel to np.float32.

    :param image: image to be convolved
    :type image: 2D ndarray, float32
    :param kernel: kernel to be used in the convolution
    :type kernel: 2D ndarray, float32
    :param mode: output array, either valid, same, or full [same]
    :param saveMemory: if mode is not full memory can be saved by making smaller zero padding
    :type saveMemory: bool

    :return: convolved image
    :rtype: ndarray, float32
    """
    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)

    #FFT padding size (next power two)
    sw = np.array(image.shape)
    sf = np.array(kernel.shape)

    if saveMemory:
        #maximum of the two
        sfft = [max(sw[0], sf[0]) - 1, max(sw[1], sf[1]) - 1]
    else:
        #nominal numerical recipes type zero padding
        sfft = sw + sf - 1

    sfft_gpu = (2 ** np.ceil(np.log2(sfft)))
    sfft_gpu = (int(sfft_gpu[0]), int(sfft_gpu[1]))

    #print 'Image size:', image.shape
    #print 'Kernel size:', kernel.shape
    #print 'Zero padded FFT size:', sfft_gpu

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

    #clean up the GPU memory
    tmp.gpudata.free()
    del fft_gpu
    del u_gpu

    return result


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

    x_cropped_gpu = cua.empty(tuple((int(size[0]),int(size[1]))), np.float32)

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
        #array = np.array(array).astype(np.float32)
        array_gpu = cua.to_gpu(array)
        #array_gpu = cua.to_gpu_async(array)
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