# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax>=0.6.2",
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

import pathlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.typing import ArrayLike

plt.rcParams.update(
    {
        "figure.figsize": (6, 6),  # figure size
        "figure.dpi": 150,  # figure dots per inch
        "text.usetex": True,  # LaTeX rendering for text
        "axes.linewidth": 1.0,  # line width for axes
        "lines.linewidth": 1.0,  # line width for plot lines
        "lines.marker": None,  # no markers on the lines
        "font.family": "serif",  # use serif fonts
        "axes.labelsize": 15,  # axis label font size
        "axes.titlesize": 20,  # axis title font size
        "xtick.labelsize": 12,  # x-axis tick label size
        "ytick.labelsize": 12,  # y-axis tick label size
        "legend.fontsize": 12,  # legend font size
        "figure.autolayout": True,  # enable tight layout
    }
)


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


def DFM(t: int, y: ArrayLike, args: ArrayLike) -> jax.Array:
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
    args : ArrayLike
        The variables and parameters of the ODE system.

    Returns
    -------
    jax.Array
        The resultant population and state resources.
    """
    # population, state resources
    N, S = y
    # population growth rate, taxation rate,
    # expenditure rate, initial carrying
    # capacity, carrying capacity increase
    # capacity from state resources,
    # initial state resources
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


# %% LOAD STYLE SHEET IF AVAILABLE

base_style_path = pathlib.Path("../assets/styles")
style = "fig_replication"
style_path = base_style_path / (style + ".mplstyle")
if style_path.exists():
    plt.style.use(str(style_path))


t0 = 0
t1 = 500
dt0 = 1
init_S = 0.0
init_N = 0.5
init_p = 1
init_s = 10
init_k = 1
max_k = 4
c = 3
r = 0.02
beta = [0.0, 0.1, 0.25, 0.4]

#
