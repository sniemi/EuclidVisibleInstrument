"""
Self-Organizing Maps

http://www.pymvpa.org/generated/mvpa2.mappers.som.SimpleSOMMapper.html#mvpa2.mappers.som.SimpleSOMMapper
"""
import matplotlib.pyplot as plt
import numpy as np
import pyfits as pf
#from mvpa2.suite import SimpleSOMMapper
from support import files as fileIO
import glob


def example():
    colors = np.array([[0., 0., 0.],
                       [0., 0., 1.],
                       [0., 0., 0.5],
                       [0.125, 0.529, 1.0],
                       [0.33, 0.4, 0.67],
                       [0.6, 0.5, 1.0],
                       [0., 1., 0.],
                       [1., 0., 0.],
                       [0., 1., 1.],
                       [1., 0., 1.],
                       [1., 1., 0.],
                       [1., 1., 1.],
                       [.33, .33, .33],
                       [.5, .5, .5],
                       [.66, .66, .66]])

    #instantiate mapper
    #use 100 x 100 units and train the network with 1000 iterations
    som = SimpleSOMMapper((50, 50), 1000, learning_rate=0.05)

    #train the mapper with the previously defined dataset
    som.train(colors)

    #find out which coordinates the initial training prototypes were mapped to
    mapped = som(colors)

    #plot
    # store the names of the colors for visualization later on
    color_names = ['black', 'blue', 'darkblue', 'skyblue', 'greyblue', 'lilac', 'green', 'red',
                   'cyan', 'violet', 'yellow', 'white', 'darkgrey', 'mediumgrey', 'lightgrey']

    plt.figure()
    plt.title('Color SOM')
    plt.imshow(som.K, origin='lower')

    # SOM's kshape is (rows x columns), while matplotlib wants (X x Y)
    for i, m in enumerate(mapped):
        plt.text(m[1], m[0], color_names[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))

    plt.show()


def simplePSFexample():
    files = glob.glob('cutout*.fits')

    all = []
    sides = []
    for f in files:
        #load data
        data = pf.getdata(f)
        sides.append(data.shape[0])
        #flatten to a 1D array and save the info
        all.append(np.ravel(data))

    all = np.asarray(all)

    #save the mean
    mean = np.mean(all, axis=0).reshape(sides[0], sides[0])
    mean /= np.max(mean)
    fileIO.writeFITS(mean, 'mean.fits', int=False)

    som = SimpleSOMMapper((100, 100), 100)
    som.train(all)

    plt.figure()
    plt.title('Color SOM')
    plt.imshow(som.K, origin='lower')
    plt.savefig('mapped.pdf')

    fileIO.writeFITS(som.K, '/mapped.fits', int=False)


def testSOMs():
    from sklearn import datasets
    from minisom import MiniSom

    d = datasets.load_iris()
    data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, d['data']) # data normalization

    som = MiniSom(7, 7, 4, sigma=1.0, learning_rate=0.5)

    som.random_weights_init(data)
    print("Training...")
    som.train_random(data, 1000) # random training
    print("\n...ready!")

    ### Plotting the response for each pattern in the iris dataset ###
    from pylab import plot,axis,show,pcolor,colorbar,bone
    bone()
    pcolor(som.distance_map().T) # plotting the distance map as background
    colorbar()
    t = d['target']
    # use different colors and markers for each label
    markers = ['o','s','D']
    colors = ['r','g','b']
    for cnt,xx in enumerate(data):
     w = som.winner(xx) # getting the winner
     # palce a marker on the winning position for the sample xx
     plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',
        markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)
    axis([0,som.weights.shape[0],0,som.weights.shape[1]])
    show() # show the figure


if __name__ == '__main__':
    #example()
    #simplePSFexample()
    testSOMs()