# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax>=0.6.2",
# .    "diffrax>=0.7.0",
#     "matplotlib>=3.10.3",
# ]
# ///


"""
Recreates figure one, "Dynamics of model (1)", pp. 06/21,
where model one is the Demographic Fiscal Model (DFM) from
Historical Dynamics (Turchin, 2003, pp.121) as stated in the
2024 paper (The Demographic-Wealth model for
cliodynamics) by Wittmann and Kuehn.
"""

import diffrax
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from jax.typing import ArrayLike

mpl.rcParams["figure.figsize"] = (10, 5)  # figure size
mpl.rcParams["figure.dpi"] = 150  # figure dots per inch
mpl.rcParams["text.usetex"] = True  # LaTeX rendering for text
mpl.rcParams["axes.linewidth"] = 1.0  # line width for axes
mpl.rcParams["font.family"] = "serif"  # use serif fonts
mpl.rcParams["axes.labelsize"] = 15  # axis label font size
mpl.rcParams["axes.titlesize"] = 20  # axis title font size
mpl.rcParams["xtick.labelsize"] = 12  # x-axis tick label size
mpl.rcParams["ytick.labelsize"] = 12  # y-axis tick label size
mpl.rcParams["legend.fontsize"] = 12  # legend font size
# mpl.rcParams["figure.autolayout"] = True       # enable tight layout
mpl.rcParams["axes.grid"] = True  # enable grid
mpl.rcParams["grid.color"] = "lightgray"  # grid line color
mpl.rcParams["grid.linestyle"] = "-"  # grid line style
mpl.rcParams["grid.linewidth"] = 0.7  # grid line width
mpl.rcParams["grid.alpha"] = 0.7  # grid transparency


def k(S: float, init_k: int, c: int, init_s: int) -> float:
    """
    A state's carrying capacity, as used in the DFM.
    Determined in part by the state's current resources,
    it's initial carrying capacity, how well the state can
    convert resources into carrying capacity, and the
    state's initial resources.

    Parameters
    ----------
    S : float
        The state's accumulated resources.
    init_k : int
        The state's initial carrying capacity.
    c : int
        How well the state can convert resources into
        carrying capacity, i.e. k_max - k_init.
    init_s : int
        The state's initial resources.

    Returns
    -------
    float
        The state's carrying capacity.
    """
    return init_k + (c * (S / (init_s + S)))


def DFM(t: int, y: ArrayLike, args: tuple) -> jax.Array:
    """
    The Demographic Fiscal Model (DFM), which models a
    state's population and accumulated resources, as
    determined by its initial population and initial
    accumulated resources, along with its rate of
    population growth, per capita taxation rate, ability
    to convert resources into carrying capacity, and its
    per capita expenditures.

    Parameters
    ----------
    t : int
        The current point in time.
    y : ArrayLike
        The current population and accumulated state
        resources.
    args : tuple
        The variables and parameters of the ODE system.

    Returns
    -------
    jax.Array
        The resultant population and state resources.
    """
    N, S = y
    r, init_rho, beta, init_k, c, init_s = args
    dN = jnp.where(
        S >= 0.0,
        r * N * (1 - (N / k(S, init_k, c, init_s))),
        r * N * (1 - (N / k(0.0, init_k, c, init_s))),
    )
    dS = jnp.where(
        S >= 0.0,
        (init_rho * N * (1 - (N / k(S, init_k, c, init_s)))) - (beta * N),
        0.0,
    )
    return jnp.array([dN, dS])


def main():
    # PARAMETERS

    model_name = "DFM"
    t0 = 0  # initial time of experiment
    t1 = 500  # final time of experiment
    dt0 = 1  # initial step size for ODE solver
    init_S = 0.0  # added state resources
    init_N = 0.5  # initial population
    init_rho = 1  # taxation rate
    init_s = 10  # initial state resources
    init_k = 1  # initial carrying capacity (CC)
    max_k = 4  # maximum carrying capacity (CC)
    c = max_k - init_k  # maximum possible CC gain via increasing S
    r = 0.02  # population growth rate
    betas = [0.0, 0.1, 0.25, 0.4]  # expenditure rates

    # ODE SOLVER AND PLOTTING SETUP

    y0 = jnp.array([init_N, init_S])
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, t1 - t0))
    solver = diffrax.Tsit5()
    term = diffrax.ODETerm(DFM)
    figure, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set_title(f"{model_name}: Population Change")
    axes[0].set_ylabel(r"$N$", rotation=90)
    axes[0].set_xlabel("t")
    axes[1].set_title(f"{model_name}: State Resources")
    axes[1].set_ylabel(r"$S$", rotation=90)
    axes[1].set_xlabel("t")

    # ODE SOLVING AND PLOTTING

    beta_colors = ["black", "green", "blue", "red"]
    for i, beta in enumerate(betas):
        args = (r, init_rho, beta, init_k, c, init_s)
        sol = diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0, args=args, saveat=saveat
        )
        N, S = sol.ys.T
        ts = sol.ts
        axes[0].plot(ts.tolist(), N.tolist(), color=beta_colors[i])
        axes[1].plot(ts.tolist(), S.tolist(), color=beta_colors[i])
    # these must be defined after plotting
    axes[0].set_xlim(left=0.0)
    axes[0].set_ylim(bottom=0.0)
    axes[1].set_xlim(left=0.0)
    axes[1].set_ylim(bottom=0.0)

    # FIGURE SHOWING AND SAVING
    figure.savefig("figure_01.png")
    plt.show()


if __name__ == "__main__":
    main()
