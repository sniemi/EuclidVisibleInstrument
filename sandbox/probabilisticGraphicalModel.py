"""
http://en.wikipedia.org/wiki/Bayesian_network
"""
from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
import daft


def weakLensing():
    pgm = daft.PGM([4.7, 2.35], origin=[-1.35, 2.2])

    pgm.add_node(daft.Node("Omega", r"$\Omega$", -1, 4))
    pgm.add_node(daft.Node("rho", r"$\rho$", 0, 4))
    pgm.add_node(daft.Node("obs", r"$\epsilon^{\mathrm{obs}}_n$", 1, 4, observed=True))
    pgm.add_node(daft.Node("alpha", r"$\alpha$", 3, 4))
    pgm.add_node(daft.Node("true", r"$\epsilon^{\mathrm{true}}_n$", 2, 4))
    pgm.add_node(daft.Node("sigma", r"$\sigma_n$", 1, 3))
    pgm.add_node(daft.Node("Sigma", r"$\Sigma$", 0, 3))
    pgm.add_node(daft.Node("x", r"$x_n$", 2, 3, observed=True))

    pgm.add_plate(daft.Plate([0.5, 2.25, 2, 2.25], label=r"galaxies $n$"))

    pgm.add_edge("Omega", "rho")
    pgm.add_edge("rho", "obs")
    pgm.add_edge("alpha", "true")
    pgm.add_edge("true", "obs")
    pgm.add_edge("x", "obs")
    pgm.add_edge("Sigma", "sigma")
    pgm.add_edge("sigma", "obs")

    pgm.render()

    pgm.figure.savefig("weaklensing.pdf")


if __name__ == '__main__':
    weakLensing()