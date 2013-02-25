import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


fig = plt.figure()
fig = plt.figure()
ax1 = fig.add_subplot(121, aspect='equal', autoscale_on=False, xlim=(-0, 2), ylim=(-1.5, 1.5))
ax2 = fig.add_subplot(122, aspect='equal', autoscale_on=False, xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
line, = ax1.plot([], [], lw=2)
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
scatter = ax2.scatter([], [], s=15)


def init():
    # initialization function: plot the background of each frame
    line.set_data([], [])
    scatter.set_offsets([])
    time_text.set_text(' ')
    return line, scatter, time_text


def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    xs = 2*np.random.rand(100) - 1
    line.set_data(x, y)
    scatter.set_offsets(xs)
    time_text.set_text('%i seconds' % i)
    return line, scatter, time_text

#note that the frames defines the number of times animate functions is being called
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=500, interval=20, blit=True)
anim.save('example.mp4', fps=30)