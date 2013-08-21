"""
Some tests with CUDA
"""
import numpy as np


def deviceInfo():
    import pycuda.driver as drv

    drv.init()
    print "%d device(s) found." % drv.Device.count()

    for ordinal in range(drv.Device.count()):
        dev = drv.Device(ordinal)
        print "Device #%d: %s" % (ordinal, dev.name())
        print "  Compute Capability: %d.%d" % dev.compute_capability()
        print "  Total Memory: %s KB" % (dev.total_memory()//(1024))
        atts = [(str(att), value)
                for att, value in dev.get_attributes().iteritems()]
        atts.sort()

        for att, value in atts:
            print "  %s: %s" % (att, value)


def firstExample():
    """
    First CUDA example, simply multiply two arrays together in two different ways.

    :return: None
    """
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray

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


def fromSourceFile():
    import numpy as np
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    #random data
    np.random.seed(1)
    a = np.random.randn(4,4)
    a = a.astype(np.float32)

    #read code and get function
    mod = SourceModule(open('simple.cu').read())
    func = mod.get_function("doublify")

    #allocate memory on the GPU
    a_gpu = cuda.mem_alloc(a.nbytes)

    #transfer to the GPU memory
    cuda.memcpy_htod(a_gpu, a)

    #execute
    func(a_gpu, block=(4,4,1))

    #collect results
    a_doubled = np.empty_like(a)
    cuda.memcpy_dtoh(a_doubled, a_gpu)

    print a_doubled
    print a_doubled / (a*2)


def simpleFourierTest2D(N=2048):
    """
    Using PyFFT to call CUDA.

    :return:
    """
    from pyfft.cuda import Plan
    import pycuda.driver as cuda
    from pycuda.tools import make_default_context
    import pycuda.gpuarray as gpuarray
    import time

    cuda.init()
    context = make_default_context()
    stream = cuda.Stream()

    plan = Plan((N, N), dtype=np.complex64, stream=stream)
    x = np.ones((N, N), dtype=np.complex64)

    x_gpu = gpuarray.to_gpu(x)

    plan.execute(x_gpu)
    res = x_gpu.get()
    plan.execute(x_gpu, inverse=True)
    result = x_gpu.get()
    context.pop()

    error = np.abs(np.sum(np.abs(x) - np.abs(result)) / x.size)
    #print 'Error:', error

    #Single precision
    print 'Array size %i x %i' % (N, N)
    print 'Single Precisions'
    x = np.random.random((N, N))
    x = x.astype(np.complex64)

    start = time.time()
    cuda.init()
    context = make_default_context()
    stream = cuda.Stream()

    plan = Plan((N, N), dtype=np.complex64, stream=stream, fast_math=True)

    x_gpu = gpuarray.to_gpu(x)
    plan.execute(x_gpu)
    result = x_gpu.get()
    context.pop()
    end = time.time()
    cudatime = end - start

    #numpy
    start = time.time()
    xf = np.fft.fft2(x)
    end = time.time()
    numpytime = end - start

    print 'Same to 1e-2?'
    print np.testing.assert_allclose(xf, result, rtol=1e-2)
    print 'Numpy time', numpytime
    print 'CUDA time', cudatime

    #Double precision
    print '\n\nDouble Precision'
    x = np.random.random((N, N))
    x = x.astype(np.complex128)

    start = time.time()

    cuda.init()
    context = make_default_context()
    stream = cuda.Stream()

    plan = Plan((N, N), dtype=np.complex128, stream=stream, fast_math=True)

    x_gpu = gpuarray.to_gpu(x)
    plan.execute(x_gpu)
    result = x_gpu.get()
    context.pop()

    end = time.time()
    cudatime = end - start

    #numpy
    start = time.time()
    xf = np.fft.fft2(x)
    end = time.time()
    numpytime = end - start

    print 'Same to 1e-7?'
    print np.testing.assert_allclose(xf, result, rtol=1e-7)
    print 'Numpy time', numpytime
    print 'CUDA time', cudatime


def simpleConvolutionTest(N=1024, test=True):
    """
    Note that the CUDA result differs. There is a complication with zero-padding.


    :param N:
    :param test:
    :return:
    """
    import pyfits as pf
    from scipy import signal
    from pyfft.cuda import Plan
    import pycuda.driver as cuda
    from pycuda.tools import make_default_context
    import pycuda.gpuarray as gpuarray
    import time

    np.random.seed(1)

    in1 = np.random.random((N, N))

    if test:
        in2 = np.random.random((N, N))
    else:
        in2 = pf.getdata('/Users/sammy/EUCLID/vissim-python/data/psf12x.fits')

    in1 = in1.astype(np.complex64)
    in2 = in2.astype(np.complex64)

    #scipy
    start = time.time()
    conv1 = signal.fftconvolve(in1, in2, mode='full').real
    end = time.time()
    scipytime = end - start

    #numpy with zero padding
    r1, c1 = in1.shape
    r2, c2 = in2.shape
    r = 2*max(r1, r2)
    c = 2*max(c1, c2)
    pr2 = int(np.log(r)/np.log(2.) + 1.)
    pc2 = int(np.log(c)/np.log(2.) + 1.)
    rOrig = r
    cOrig = c
    r = 2**pr2
    c = 2**pc2
    start = time.time()
    fftimage = np.fft.fft2(in1, s=(r,c)) * np.fft.fft2(in2, s=(r,c))
    conv2 = np.fft.ifft2(fftimage)[:rOrig-1, :cOrig-1].real
    end = time.time()
    numpytime = end - start

    print 'Scipy %f and Numpy %f seconds' % (scipytime, numpytime)
    print np.testing.assert_allclose(conv1, conv2, rtol=1e-1)

    #CUDA
    fftimage = np.fft.fft2(in1) * np.fft.fft2(in2)
    conv3 = np.fft.ifft2(fftimage).real

    start = time.time()

    cuda.init()
    context = make_default_context()

    plan = Plan(in1.shape, dtype=np.complex64)

    x_gpu1 = gpuarray.to_gpu(in1)
    x_gpu2 = gpuarray.to_gpu(in2)
    plan.execute(x_gpu1)
    plan.execute(x_gpu2)
    tmp = x_gpu1 * x_gpu2
    plan.execute(tmp, inverse=True)
    result = tmp.get()[:rOrig-1, :cOrig-1].real
    context.pop()

    end = time.time()
    cudatime = end - start

    print 'CUDA %f' % cudatime
    print np.testing.assert_allclose(conv3, result, rtol=1e-1)


def simpleConvolution(mode='valid'):
    """
    Simple convolution test with random data. Tests if the GPU convolution
    returns the same result as SciPy.signal.fftconvolve. This example
    uses single precision and an image that is about 2k x 2k and a kernel
    that is about 200 x 200.

    :param mode: the resulted convolution area (valid, same, full)
    :type mode: str

    :return: None
    """
    import pycuda.autoinit
    import pycuda.gpuarray as cua
    import numpy as np
    import scipy
    from scipy import signal
    import time

    # Load VMDB libraries
    import gputools
    import imagetools
    import olaGPU as ola

    np.random.seed(123)

    #data
    x = np.random.random((2099, 2100)).astype(np.float32)  #don't make the array too large, not enough GPU memory
    scipy.misc.imsave('originalSimple.jpg', np.log10(x))

    #kernel
    kernel = np.random.random((299, 299)).astype(np.float32)
    kernel /= np.sum(kernel)
    scipy.misc.imsave('kernelSimple.jpg', np.log10(kernel))

    print x.shape, kernel.shape

    x_gpu = cua.to_gpu(x)

    sx = x.shape
    csf = (5,5)
    overlap = 0.5

    fs = np.tile(kernel, (np.prod(csf), 1, 1))

    winaux = imagetools.win2winaux(sx, csf, overlap)

    print "-------------------"
    print "Create CUDA Kernel"
    start = time.clock()
    F = ola.OlaGPU(fs, sx, mode=mode, winaux=winaux)

    print "Compute Convolution with the GPU using FFTs"
    yF_gpu = F.cnv(x_gpu)

    print "Copy results to CPU"
    result = yF_gpu.get()
    cutime = time.clock()-start

    print "Time elapsed: %.4f" % cutime
    print "-------------------"

    print "SciPy FFT convolution on CPU"
    start = time.clock()
    conv = signal.fftconvolve(x, kernel, mode=mode)
    sptime = time.clock()-start
    print "Time elapsed: %.4f" % sptime
    print "-------------------"
    print 'CUDA is a factor of %.2f faster' % (sptime / cutime)

    #save images
    scipy.misc.imsave('convolvedCUDASimple.jpg', np.log10(result))
    scipy.misc.imsave('convolvedSciPySimple.jpg', np.log10(conv))

    print '\n\n\nShapes:', result.shape, conv.shape
    print 'Max values:', np.max(result), np.max(conv)

    print '\n\nDifference:'
    if 'full' or 'same' in mode:
        print '> 1e-5?'
        print np.testing.assert_allclose(result[100:-100, 100:-100], conv[100:-100, 100:-100], rtol=1e-5)
    else:
        print '> 1e-6?'
        print np.testing.assert_allclose(result, conv, rtol=1e-6)


def galaxyConvolution(mode='same'):
    import pycuda.autoinit
    import pycuda.gpuarray as cua
    import numpy as np
    import pyfits as pf
    import scipy
    from scipy import signal
    from scipy import ndimage
    import time

    # Load VMDB libraries
    import gputools
    import imagetools
    import olaGPU as ola

    np.random.seed(123)

    #data
    x = pf.getdata('/Users/sammy/EUCLID/vissim-python/objects/galaxy37.fits')
    #x = pf.getdata('/Users/sammy/EUCLID/vissim-python/objects/galaxy57.fits')
    #x = pf.getdata('/Users/sammy/EUCLID/vissim-python/objects/galaxy22.fits')
    x = ndimage.zoom(x, 4.0, order=0)  #oversampling = 12 leads to out of memory error...
    x[x <= 0] = 1e-8
    x /= np.max(x)
    x *= 4000.
    x = x.astype(np.float32)
    scipy.misc.imsave('original.jpg', np.log10(x))

    #kernel
    kernel = pf.getdata('/Users/sammy/EUCLID/vissim-python/data/psf4x.fits')
    kernel /= np.sum(kernel)
    kernel = kernel.astype(np.float32)
    scipy.misc.imsave('kernel.jpg', np.log10(kernel))

    print x.shape, kernel.shape

    x_gpu = cua.to_gpu(x)

    sx = x.shape
    csf = (5,5)
    overlap = 0.5

    fs = np.tile(kernel, (np.prod(csf), 1, 1))
    fs_gpu = cua.to_gpu(fs)

    winaux = imagetools.win2winaux(sx, csf, overlap)

    print "-------------------"
    print "Create Kernel"
    start = time.clock()
    F = ola.OlaGPU(fs, sx, mode=mode, winaux=winaux)

    print "Compute Convolution "
    yF_gpu = F.cnv(x_gpu)
    print "Copy to CPU "
    result = yF_gpu.get()
    print "Time elapsed: %.4f" % (time.clock()-start)

    #other way around
    X = ola.OlaGPU(x_gpu, kernel.shape, mode=mode, winaux=winaux)
    yX_gpu = X.cnv(fs_gpu)
    result2 = yX_gpu.get()

    print "-------------------"
    print "SciPy FFT convolution "
    start = time.clock()
    conv = signal.fftconvolve(x, kernel, mode=mode)
    print "Time elapsed: %.4f" % (time.clock()-start)
    print "-------------------"

    #save images
    r = result.copy()
    r[r <= 0.] = 1e-5
    c = conv.copy()
    c[c <= 0.] = 1e-5
    scipy.misc.imsave('convolvedCUDA.jpg', np.log10(r))
    scipy.misc.imsave('convolvedSciPy.jpg', np.log10(c))

    print 'Shapes:', result.shape, conv.shape
    print 'Max:', np.max(result), np.max(conv)

    print 'Differences:'
    if 'full' in mode:
        print '> 1e-2?'
        print np.testing.assert_allclose(result[100:-100, 100:-100], conv[100:-100, 100:-100], rtol=1e-2)
    else:
        print  '> 1e-4?'
        print np.testing.assert_allclose(result, conv, rtol=1e-4)


def Test2DFFTmultiprocessing():
    import numpy
    import scipy.misc
    import multiprocessing

    from pyfft.cuda import Plan
    from pycuda.tools import make_default_context
    import pycuda.tools as pytools
    import pycuda.gpuarray as garray
    import pycuda.driver as drv


    class GPUMulti(multiprocessing.Process):
        def __init__(self, number, input_cpu, output_cpu):
            multiprocessing.Process.__init__(self)
            self.number = number
            self.input_cpu = input_cpu
            self.output_cpu = output_cpu

        def run(self):
            drv.init()
            a0=numpy.zeros((p,),dtype=numpy.complex64)
            self.dev = drv.Device(self.number)
            self.ctx = self.dev.make_context()
    #TO VERIFY WHETHER ALL THE MEMORY IS FREED BEFORE NEXT ALLOCATION (THIS DOES NOT HAPPEN IN MULTITHREADING)
            print drv.mem_get_info()
            self.gpu_a = garray.empty((self.input_cpu.size,), dtype=numpy.complex64)
            self.gpu_b = garray.zeros_like(self.gpu_a)
            self.gpu_a = garray.to_gpu(self.input_cpu)
            plan = Plan(a0.shape,context=self.ctx)
            plan.execute(self.gpu_a, self.gpu_b, batch=p/m)
            self.temp = self.gpu_b.get()
            self.output_cpu.put(self.temp)
            self.output_cpu.close()
            self.ctx.pop()
            del self.gpu_a
            del self.gpu_b
            del self.ctx

    p = 8192 # INPUT IMAGE SIZE (8192 * 8192)
    m = 4     # TO DIVIDE THE INPUT IMAGE INTO 4* (2048 * 8192) SIZED IMAGES (Depends on the total memory of your GPU)
    trans = 2 # FOR TRANSPOSE-SPLIT (TS) ALGORITHM WHICH loops 2 times

    #INPUT IMAGE (GENERATE A 2d SINE WAVE PATTERN)
    p_n = 8000 # No. OF PERIODS OF SINE WAVES
    x=numpy.arange(0,p_n,float(p_n)/float(p))
    a_i = 128 + 128 * numpy.sin(2*numpy.pi*x)
    a2 = numpy.zeros([p,p],dtype=numpy.complex64)
    a2[::]=a_i
    #scipy.misc.imsave("sine.bmp",numpy.absolute(a2)) #TEST THE GENERATION OF INPUT IMAGE

    #INITIALISE THE VARIABLES
    a2_1 = numpy.zeros([m,p*p/m],dtype = numpy.complex64) #INPUT TO THE GPU (1d ARRAY)
    #VERY IMPORTANT
    output_cpu  = multiprocessing.Queue() #STORE RESULT IN GPU (MULTIPROCESSING DOES NOT ALLOW SHARING AND HENCE THIS IS NEEDED FOR COMMUNICATION OF DATA)

    b2pa = numpy.zeros([p/m,p,m],dtype = numpy.complex64) #OUTPUT FROM GPU
    b2_a = numpy.zeros([p,p],dtype = numpy.complex64)     #RESHAPED (8192*8192) OUTPUT

    #NOW WE ARE READY TO KICK START THE GPU

    # THE NO OF GPU'S PRESENT (CHANGE ACCORDING TO THE No.OF GPUS YOU HAVE)
    num = 1 # I KNOW THIS IS A BAD PRACTISE, BUT I COUNDN'T FIND ANY OTHER WAY(INIT CANNOT BE USED HERE)

    #THE TRANSPOSE-SPLIT ALGORITHM FOR FFT
    for t in range (0,trans):
        for i in range (m):
            a2_1[i,:] = a2[i*p/m:(i+1)*p/m,:].flatten()#DIVIDE AND RESHAPE THE INPUT IMAGE INTO 1D ARRAY

        for j in range (m/num):
            gpu_multi_list = []

    #CREATE AND START THE MULTIPROCESS
            for i in range (num):
                gpu_multi = GPUMulti(i,a2_1[i+j*num,:],output_cpu) #FEED THE DATA INTO THE GPU
                gpu_multi_list.append(gpu_multi)
                gpu_multi.start()#THERE YOU GO

    #COLLECT THE OUTPUT FROM THE RUNNING MULTIPROCESS AND RESHAPE
            for gpu_pro in gpu_multi_list:
                temp_b2_1 = output_cpu.get(gpu_pro)
                b2pa[:,:,gpu_pro.number+j*num] = numpy.reshape(temp_b2_1,(p/m,p))
            gpu_multi.terminate()

    #RESHAPE AGAIN TO (8192 * 8192) IMAGE
        for i in range(m):
            b2_a[i*p/m:(i+1)*p/m,:] = b2pa[:,:,i]

    print b2_a


if __name__ == "__main__":
    #deviceInfo()
    #firstExample()
    #fromSourceFile()
    #simpleFourierTest2D()

    #simpleConvolution()
    galaxyConvolution()