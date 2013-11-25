import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

def one():

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


def two():
    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(7, 6.5)

    ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
    patch = plt.Circle((5, -5), 0.75, fc='y')

    def init():
        patch.center = (5, 5)
        ax.add_patch(patch)
        return patch,

    def animate(i):
        x, y = patch.center
        x = 5 + 3 * np.sin(np.radians(i))
        y = 5 + 3 * np.cos(np.radians(i))
        patch.center = (x, y)
        return patch,

    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=360,
                                   interval=20,
                                   blit=True)

    plt.show()