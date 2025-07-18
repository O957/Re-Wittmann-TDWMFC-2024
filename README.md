# Partial Replication Of (The Demographic-Wealth Model For Cliodynamics)

> [!CAUTION]
> This project is currently ACTIVE but not FINISHED. As such, there may be parts of this repository that do not make much sense or that are broken!

## [This Repository](repository)

__What is this repository?__

This repository is a partial replication in Python of the 2024 paper (The Demographic-Wealth model for cliodynamics) by Wittmann and Kuehn, abbreviated hereafter as Wittmann_TDWMFC_2024.

Specifically, this reproduction entails that:

* [ ] Code reproductions of all figures from Wittmann_TDWMFC_2024.
* [ ] A `streamlit` application for running experiments with the demographic fiscal model (henceform abbreviated as DFM) and demographic wealth models (henceforth abbreviated as DWM).
* [ ] Descriptions of 5 additional DWM model configurations.
* [ ] Proofs of theorems 4.1 and 4.2 verified using Lean.

__What background information is needed?__

The suggested know-requisites for engaging with this repository include:

* Familiarity with Peter Turchin's _Historical Dynamics_.
  * Find key equations in `./assets/misc/equations.md`.
* Familiarity with the paper this repository targets.
  * Find a PDF in `./assets/papers`.
  * Find key equations in `./assets/misc/equations.md`.
* Familiarity with the Python programming language.
* Familiarity with nonlinear dynamics.

__What is in this repository?__

* The folder `.github` contains:
  * GitHub Actions workflows (see [here]()).
  * Code owners file (see [here]()).
* The folder `assets` contains:
  * Original figures from Wittmann_TDWMFC_2024 (see [here]()).
  * A PDF of Wittmann_TDWMFC_2024 (see [here]()).
  * Matplotlib styles used for plotting (see [here]()).
  * Decisions relevant to this repository (see [here]()).
  * Equations relevant to this repository (see [here]()).
  * Glossary terms relevant to this repository (see [here]()).
  * Online resources relevant to this repository (see [here]()).
  * A project roadmap for this repository (see [here]()).
  * A feature list for this repository (see [here]()).
* The folder `src` contains:
  * Python scripts for each figure in Wittmann_TDWMFC_2024 (see [here]()).
  * The `streamlit` application for experiments involving the DFM and DWM (see [here]()).

## [Usage](#usage)

__How can this repository can be used?__

This repository supports:

1. Running new DWM and DFM experiments via the `streamlit` application.
2. Reproducing or modifying the original figures from Wittmann_TDWMFC_2024.
3. Verifying the proofs for Theorem 4.1. and 4.2.

For (1), head to the `streamlit` application [here]() and enter desired variable values.

For (2), either:

* Download these files manually, to run or view locally..
  * [ ] Figure 1 [Code](), [Output]()
  * [ ] Figure 2 [Code](), [Output]()
  * [ ] Figure 3 [Code](), [Output]()
  * [ ] Figure 4 [Code](), [Output]()
  * [ ] Figure 5 [Code](), [Output]()
  * [ ] Figure 6 [Code](), [Output]()
  * [ ] Figure 7 [Code](), [Output]()
  * [ ] Figure 8 [Code](), [Output]()
  * [ ] Figure 9 [Code](), [Output]()
  * [ ] Figure 10 [Code](), [Output]()
  * [ ] Figure 11 [Code](), [Output]()
* Clone this repository and run code from within:
  * `git clone ...`
  * `cd src`
  * `python3 <target figure>.py`.

For (3): (pending)


## [Motivation](#motivation)

__Why does this repository exist?__

This repository exists because:

* I find cliodynamics and nonlinear dynamics to be both interesting and fulfilling enterprises.
* I expected engaging with Wittmann_TDWMFC_2024 would further my understanding of cliodynamics and nonlinear dynamics.
* I am interested in reproducing / replicating scientific enterprises, for personal development and to further human knowledge.

> [!NOTE]
> This project represents my first public reproduction / replication effort. Subsequent public efforts I take in this regard may be structured differently. Even though this project has been declared FINISHED, I may sync it with future structural standards I adopt for reproduction / replication efforts.

## [Original Paper](#original-paper)

__Citation__:

> Wittmann, Lukas, and Christian Kuehn. "The Demographic-Wealth model for cliodynamics." Plos one 19, no. 4 (2024): e0298318.

__Citation BibTeX__:

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

__Relevant Links__

* The paper online (<https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0298318&type=printable>).
* The paper's GS citations (<https://scholar.google.com/scholar?cites=4147056941143982529&as_sdt=5,44&sciodt=0,44&hl=en>).

__Paper Abstract__:

<details markdown=1>

> Cliodynamics is a still a relatively new research area with the purpose of investigating and modelling historical processes. One of its first important mathematical models was proposed by Turchin and called “Demographic-Fiscal Model” (DFM). This DFM was one of the first and is one of a few models that link population with state dynamics. In this work, we propose a possible alternative to the classical Turchin DFM, which contributes to further model development and comparison essential for the field of cliodynamics. Our “Demographic-Wealth Model” (DWM) aims to also model link between population and state dynamics but makes different modelling assumptions, particularly about the type of possible taxation. As an important contribution, we employ tools from nonlinear dynamics, e.g., existence theory for periodic orbits as well as analytical and numerical bifurcation analysis, to analyze the DWM. We believe that these tools can also be helpful for many other current and future models in cliodynamics. One particular focus of our analysis is the occurrence of Hopf bifurcations. Therefore, a detailed analysis is developed regarding equilibria and their possible bifurcations. Especially noticeable is the behavior of the so-called coexistence point. While changing different parameters, a variety of Hopf bifurcations occur. In addition, it is indicated, what role Hopf bifurcations may play in the interplay between population and state dynamics. There are critical values of different parameters that yield periodic behavior and limit cycles when exceeded, similar to the “paradox of enrichment” known in ecology. This means that the DWM provides one possible avenue setup to explain in a simple format the existence of secular cycles, which have been observed in historical data. In summary, our model aims to balance simplicity, linking to the underlying processes and the goal to represent secular cycles.

</details>

## [Contributing](#contributing)

__How may I contribute to this project?__

(please, very much so, feel free to)

* Point out errors in my code or writing.
* Propose or add to this repository additional equations or experiments that go beyond what's in Wittmann_TDWMFC_2024.
* Provide suggestions regarding my code or writing.

Some means to contribute include:

* Making [an issue](https://github.com/AFg6K7h4fhy2/Re-TDWMFC-2024/issues) in this repository.
* Making [a pull request](https://github.com/AFg6K7h4fhy2/Re-TDWMFC-2024/pulls) to this repository.
* Engaging in [a discussion thread](https://github.com/AFg6K7h4fhy2/Re-TDWMFC-2024/discussions) in this repository.
* Contacting me via email: [my username]+[@]+[pro]+[ton]+[.]+[me]
