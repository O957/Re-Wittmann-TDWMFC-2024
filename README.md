# Partial Replication Of (The Demographic-Wealth Model For Cliodynamics)

> [!CAUTION]
> This project is currently ACTIVE but not FINISHED. As such, there may be parts of this repository that do not make much sense or that are broken!

## Table Of Contents

* [This Repository](installatio)

## [This Repository](#repository)

__What is this repository?__

This repository consists of a partial replication in Python of the 2024 paper (The Demographic-Wealth model for cliodynamics) by Wittmann and Kuehn, abbreviated henceforth as Wittmann_TDWMFC_2024.

Specifically, this entails that:

* [ ] Code reproductions of all figures from Wittmann_TDWMFC_2024.
* [ ] A `streamlit` application for running experiments with the demographic fiscal model (henceform DFM) and demographic wealth models (henceforth DWM).
* [ ] Descriptions of 5 additional DWM model configurations.
* [ ] Proofs of theorems 4.1 and 4.2 verified using Lean.

__What background information is needed?__

__What is in this repository?__

* The folder `assets` contains:
* The folder `c

## [Usage](#usage)

__How can this repository can be used?__

This repository supports:

1. Running new DWM and DFM experiments via the `streamlit` application.
2. Reproduction of the original figures from Wittmann_TDWMFC_2024.

For (1), head to the `streamlit` application [here]().

For (2), either:

* Download the desired file from this repository to run locally..
* Clone this repository via `git clone ...`.


## [Motivation](#motivation)

__Why does this repository exist?__

This repository exists because:

* I am interested in cliodynamics as a field.
* I believe that engaging with Wittmann_TDWMFC_2024 helps me in this regard.
* I am interested in reproducing / replicating scientific enterprises.

## [Original Paper](#original-paper)

__Citation__:

> Wittmann, Lukas, and Christian Kuehn. "The Demographic-Wealth model for cliodynamics." Plos one 19, no. 4 (2024): e0298318.

__Bibtex__:

```
@article{wittmann2024demographic,
  title={The Demographic-Wealth model for cliodynamics},
  author={Wittmann, Lukas and Kuehn, Christian},
  journal={Plos one},
  volume={19},
  number={4},
  pages={e0298318},
  year={2024},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

__Links__

* [The paper online.](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0298318&type=printable).
* [A PDF of the paper](./assets/2024-TDWMFC-Wittmann.pdf).
* [The paper's GS citations](https://scholar.google.com/scholar?cites=4147056941143982529&as_sdt=5,44&sciodt=0,44&hl=en).

__Abstract__:

<details markdown=1>

> Cliodynamics is a still a relatively new research area with the purpose of investigating and modelling historical processes. One of its first important mathematical models was proposed by Turchin and called “Demographic-Fiscal Model” (DFM). This DFM was one of the first and is one of a few models that link population with state dynamics. In this work, we propose a possible alternative to the classical Turchin DFM, which contributes to further model development and comparison essential for the field of cliodynamics. Our “Demographic-Wealth Model” (DWM) aims to also model link between population and state dynamics but makes different modelling assumptions, particularly about the type of possible taxation. As an important contribution, we employ tools from nonlinear dynamics, e.g., existence theory for periodic orbits as well as analytical and numerical bifurcation analysis, to analyze the DWM. We believe that these tools can also be helpful for many other current and future models in cliodynamics. One particular focus of our analysis is the occurrence of Hopf bifurcations. Therefore, a detailed analysis is developed regarding equilibria and their possible bifurcations. Especially noticeable is the behavior of the so-called coexistence point. While changing different parameters, a variety of Hopf bifurcations occur. In addition, it is indicated, what role Hopf bifurcations may play in the interplay between population and state dynamics. There are critical values of different parameters that yield periodic behavior and limit cycles when exceeded, similar to the “paradox of enrichment” known in ecology. This means that the DWM provides one possible avenue setup to explain in a simple format the existence of secular cycles, which have been observed in historical data. In summary, our model aims to balance simplicity, linking to the underlying processes and the goal to represent secular cycles.

</details>
