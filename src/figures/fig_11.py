# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax>=0.6.2",
# .    "diffrax>=0.7.0",
#     "matplotlib>=3.10.3",
# ]
# ///

"""
Recreates figure eleven, "Dynamics of the
Demographic-Wealth model with limit cycle
and adapted axes", pp. 16/21 of the 2024 paper
(The Demographic-Wealth model for
cliodynamics) by Wittmann and Kuehn.
"""

import pathlib

import diffrax
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from jax.typing import ArrayLike

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "output"

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
mpl.rcParams["figure.autolayout"] = True  # enable tight layout
mpl.rcParams["axes.grid"] = True  # enable grid
mpl.rcParams["grid.color"] = "lightgray"  # grid line color
mpl.rcParams["grid.linestyle"] = "-"  # grid line style
mpl.rcParams["grid.linewidth"] = 0.7  # grid line width
mpl.rcParams["grid.alpha"] = 0.7  # grid transparency


def k(S: float, init_k: int, c: int) -> float:
    """
    A state's carrying capacity as used in
    the DWM. Determined in part by the
    state's current wealth, it's initial
    carry capacity, how well the state can
    convert wealth into carrying capacity,
    and the state's initial wealth.

    Parameters
    ----------
    S : float
        The state's wealth.
    init_k : int
        The state's initial carrying capacity.
    c : int
        How well the state can convert
        wealth into carrying capacity, i.e.
        k_max - k_init.

    Returns
    -------
    float
        The state's carrying capacity.
    """
    return init_k + (c * S)


def DWM(
    t: int,
    y: ArrayLike,
    args: ArrayLike,
) -> jax.Array:
    """
    The Demographic Wealth Model (DWM), which
    models a state's population and
    wealth, as determined by
    its initial population and its
    rate of population growth, per capita
    taxation rate, fraction of surplus gained
    through investing/expanding (using its
    wealth), ability to convert resources
    into carrying capacity, per capita
    expenditures, and negative feedback
    between population and wealth.

    Parameters
    ----------
    t : int
        The current point in time.
    y : ArrayLike
        The current population and accumulated
        state wealth.
    args : ArrayLike
        The variables and parameters of the
        ODE system.

    Returns
    -------
    jax.Array
        The resultant population and state
        resources.
    """
    # population, state resources
    N, S = y
    # population growth rate,
    # expenditure rate,
    # initial state resources,
    # expenditure rate, negative interaction
    # strength between state wealth and
    # population size, tax rate times the
    # fraction of surplus gained through
    # investing/expanding, carrying capacity
    #  increase capacity from state wealth,
    # initial carrying capacity,
    r, beta, alpha, d, g, c, init_k = args
    dN = jnp.where(
        S >= 0.0,
        (r * N * (1 - (N / k(S, init_k, c)))) - (alpha * S * (N / (d + N))),
        (r * N * (1 - (N / k(0.0, init_k, c))))
        - (alpha * 0.0 * (N / (d + N))),
    )
    dS = jnp.where(S >= 0.0, (g * S * N) - (beta * S), 0.0)
    return jnp.array([dN, dS])


def main():
    # PARAMETERS

    model_name = "DWM"
    t0 = 0  # initial time of experiment
    t1 = 500  # final time of experiment
    dt0 = 1  # initial step size for ODE solver
    init_S = 0.01  # added state resources
    init_N = 0.33  # initial population
    init_k = 1  # initial carrying capacity (CC)
    c = 0.5  # maximum possible CC gain via increasing S
    r = 0.04  # population growth rate
    d = 1  # fraction of surplus gained through
    g = 0.16  # investing/expanding, carrying capacity
    alpha = 0.006  #
    beta = 0.1  # expenditure rate

    # ODE SOLVER AND PLOTTING SETUP

    y0 = jnp.array([init_N, init_S])
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, t1 - t0))
    solver = diffrax.Tsit5()
    term = diffrax.ODETerm(DWM)
    figure, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set_title(f"{model_name}: Population Change")
    axes[0].set_ylabel(r"$N$", rotation=90)
    axes[0].set_xlabel("t")
    axes[1].set_title(f"{model_name}: State Resources")
    axes[1].set_ylabel(r"$S$", rotation=90)
    axes[1].set_xlabel("t")

    # ODE SOLVING AND PLOTTING

    color = "navy"
    args = (r, beta, alpha, d, g, c, init_k)
    sol = diffrax.diffeqsolve(
        term, solver, t0, t1, dt0, y0, args=args, saveat=saveat
    )
    N, S = sol.ys.T
    ts = sol.ts
    axes[0].plot(ts.tolist(), N.tolist(), color=color)
    axes[1].plot(ts.tolist(), S.tolist(), color=color)
    # these must be defined after plotting
    axes[0].set_xlim(left=0.0)
    axes[0].set_ylim(bottom=0.0)
    axes[1].set_xlim(left=0.0)
    axes[1].set_ylim(bottom=0.0)

    # FIGURE SHOWING AND SAVING

    save_path = OUTPUT_DIR.joinpath("figure_11.png")
    figure.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    main()
