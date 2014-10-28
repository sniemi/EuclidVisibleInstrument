import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats as s

#Note, NumPy STD:
# ddof=1 provides an unbiased estimator of the variance of the infinite population.
# ddof=0 provides a maximum likelihood estimate of the variance for normally distributed variables. 
# The standard deviation computed in this function is the square root of the estimated variance, so even with ddof=1, 
# it will not be an unbiased estimate of the standard deviation per se.


#these can be changed
gain = 3.5
rn = 4.5
bias = 300 * gain
sky = 100.
remains = 0.5
usesky = True
operator = np.rint #np.around #np.floor #np.round
ddof = 1
quadrant = 2048*2066 * remains

#calculations start
#tmp = np.random.normal(loc=0., scale=rn, size=quadrant).flatten()
tmp = s.norm.rvs(loc=0., scale=rn, size=quadrant).flatten()
skyb = s.poisson.rvs(mu=sky, size=quadrant).flatten()
if usesky:
    tmp =  tmp + skyb
tmp1 = tmp + bias
tmp2 = tmp1 / gain
data = operator(tmp2)

recovered = data * gain - bias

print 'input=', np.sqrt(rn**2 + sky)
print 'Derived='
#note that the degrees of freedom are N-1
print tmp.std(ddof=ddof), 'vs', recovered.std(ddof=ddof) , 'vs', data.std(ddof=ddof)*gain
print sp.std(tmp), 'vs', sp.std(recovered), 'vs', sp.std(data)*gain


#fit to data
locr, stdr = s.norm.fit(recovered)
loc, std = s.norm.fit(tmp)
x = np.linspace(-30+sky, 30+sky, 1000)

print loc, std
print locr, stdr
print 'size, (min, max), mean, variance, skewness, kurtosis:'
print sp.stats.describe(tmp)
print sp.stats.describe(recovered)

fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.set_title('Readout Noise Recovery')
ax2.set_title('Gain=%.1f' % gain)

ax1.hist(tmp, bins=50, normed=True, label='Input', alpha=0.5)
ax1.plot(x, s.norm(loc=loc, scale=std).pdf(x), 'g--', lw=2)
ax2.hist(recovered, bins=50, label='Output', alpha=0.5, weights=np.repeat(1./len(recovered)/gain, len(recovered)))
ax2.plot(x, s.norm(loc=loc, scale=std).pdf(x), 'g--', lw=2)
ax2.plot(x, s.norm(loc=locr, scale=stdr).pdf(x), 'r-', lw=2)

ax1.annotate('Mean = %.5f\nStd = %.3f' % (loc, std), xy=(15+sky, 0.05))
ax2.annotate('Mean = %.5f\nStd = %.3f' % (locr, stdr), xy=(15+sky, 0.05))

ax2.set_xlabel('Electrons')

ax1.set_ylim(0., 0.1)
ax2.set_ylim(0., 0.1)

ax1.legend(shadow=True)
ax2.legend(shadow=True)
plt.savefig('NoiseDistributions.pdf')
