"""

"""
import numpy as np

def CDM03(data, nt, sigma, tr, quadrant, rdose):
    """

    """
    dob = 0.0
    beta=0.6
    fwc=175000.
    vth=1.168e7
    t=1.024e-2
    vg=6.e-11
    st=5.e-6
    sfwc=730000.
    svg=1.0e-10

    ydim, xdim = data.shape
    zdim = len(nt)

    nt *= rdose

    #allocs
    no = np.zeros((ydim, zdim))
    sno = np.zeros((xdim,zdim))

    # flip data for Euclid depending on the quadrant being processed
    if quadrant == 1 or quadrant == 3:
        data = np.fliplr(data)
    if quadrant == 2 or quadrant == 3:
        data = np.flipud(data)

    #add background
    data = data + dob

    #anti-blooming
    data[data > fwc] = fwc

    #parallel direction
    alpha = t * sigma * vth * fwc**beta / 2. / vg
    g = nt * 2. * vg / fwc**beta

    for i in range(xdim):
        gamm = g * i
        for k in range(zdim):
            for j in range(ydim):
                nc = 0.

                if data[j, i] > 0.01:
                    div = (gamm[k]*data[i,j]**(beta-1.)+1.)*(1.-np.exp(-alpha[k]*data[i,j]**(1.-beta)))
                    nc = gamm[k]*data[i,j]**beta - no[j,k] / div
                    if nc < 0:
                        nc = 0.

                no[j,k] = no[j,k] + nc
                nr = no[j,k] * (1. - np.exp(-t/tr[k]))
                data[i,j] = data[i,j] - nc + nr
                no[j,k] = no[j,k] - nr


    #now serial direction
    alpha=st*sigma*vth*sfwc**beta/2./svg
    g=nt*2.*svg/sfwc**beta

    for j in range(xdim):
        gamm = g * j
        for k in range(zdim):
             if tr[k] < t:
                for i in range(ydim):
                    nc = 0.

                    if data[i,j] > 0.01:
                        nc = gamm[k]*data[i,j]**beta-sno[i,k] / (gamm[k]*data[i,j]**(beta-1.)+1.)*(1.-np.exp(-alpha[k]*data[i,j]**(1.-beta)))
                        if nc < 0.0:
                            nc = 0.

                    sno[i,k] += nc
                    nr = sno[i,k] * (1. - np.exp(-st/tr[k]))
                    data[i,j] = data[i,j] - nc + nr
                    sno[i,k] = sno[i,k] - nr


    if quadrant == 1 or quadrant == 3:
        data = np.fliplr(data)
    if quadrant == 2 or quadrant == 3:
        data = np.flipud(data)


    return data