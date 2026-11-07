# Analyzing influencer marketing strategies using network-based cellular automata

Code, simulation results and analysis scripts for the paper

> Procházka, V., Martinčík, D., Martinčíková Sojková, O., & Král, J. (2026).
> *Analyzing influencer marketing strategies using network-based cellular automata.*
> University of West Bohemia in Pilsen, Faculty of Economics.

The study asks whether a brand should spend an influencer budget on one account
with a very large audience or spread it across many smaller accounts. Four
seeding strategies with the same budget are compared in a network cellular
automaton on a real follower network (SNAP ego-Twitter, 81,306 accounts), across
levels of product adoption friction, consumer ad fatigue and campaign size, with
robustness checks on influencer pricing and network structure.

## The model in brief

* **Network.** SNAP ego-Twitter (Leskovec & McAuley, 2012), directed. A line
  `u v` of the edge list means *u follows v*, so influence flows from *v* to *u*.
  After removing self-loops and duplicate edges: 81,306 accounts and about
  1.77 million follow relations.
* **States.** Every account is a non-adopter or an adopter. Adoption is
  permanent, and adopters keep influencing their followers in every later round.
* **Update rule** (synchronous rounds). A non-adopter *i* adopts when

  R<sub>i</sub>(t) · exp(−α · E<sub>i</sub>(t)) ≥ θ<sub>i</sub>

  * R<sub>i</sub>(t): share of the accounts *i* follows that have adopted
  * θ<sub>i</sub>: individual adoption threshold (product friction), drawn from a
    normal distribution with mean μ and SD 0.10, truncated to [0.05, 0.95]
  * E<sub>i</sub>(t): number of earlier rounds in which *i* was exposed (at least
    one followed account had adopted) but did not adopt. Fatigue therefore grows
    once per exposed round, not once per active source.
* **Budget and strategies.** The budget equals the price of *K* mega-influencers
  (top 1% by followers), with price = followers<sup>γ</sup> (γ = 1 in the
  baseline). Every strategy spends approximately the same budget:
  * **Mega**: the *K* mega-influencers
  * **Micro**: random accounts from the 80th–95th follower percentile
  * **Niche**: micro accounts from one network neighbourhood (breadth-first
    search), so that their audiences overlap
  * **Random**: random accounts from all users
* **Outcomes.** A *global cascade* is final adoption by at least 10% of accounts.
  *ROSS* (return on seed spend) is the number of additional simulated adopters
  per cost unit of the model's price scale.

## Repository structure

```
simulation/                      Python / PyTorch simulation (one script per stage)
  01_pilot.py                    pilot sweep on a coarse grid
  02_extended_grid.py            extended grid used to set the final design
  03_main_experiment.py          main experiment reported in the paper
  04_robustness.py               price exponents 0.7 / 1.3, fully randomised network
  05_partial_rewiring.py         10 / 25 / 50% of edges randomised
  06_rewiring_replicates.py      optional: several independent randomised networks
data/
  raw/                           SNAP edge list (downloaded automatically, not tracked)
  results/                       simulation output, one CSV per stage + codebook
analysis/                        R analysis (open analysis/analysis.Rproj)
  R/helpers.R                    shared functions: data loading, models, Wilson intervals
  01_descriptives.R              outcome distribution, seed sets, cascade rates, ROSS
  02_strategy_models.R           odds ratios and interactions (Sections 3.2-3.4)
  03_robustness.R                price schedule and network randomisation (Section 3.5)
  04_figures.R                   Figures 2-4
  05_review_checks.R             budget-matching sensitivity, alternative models,
                                 counts behind quoted numbers
  network_diagnostics.py         structure of the original and randomised networks
  exploratory/                   step-by-step analysis logs from the design phase
  output/                        tables written by the scripts
figures/
  scripts/fig1_seeding_strategies.py   Figure 1 (schematic on a generated network)
  paper/                         figures used in the paper
LICENSE                          MIT License
```

The codebook of the result files is in [`data/results/README.md`](data/results/README.md).

## Reproducing the results

### 1. Simulation (Python)

Requires Python 3.10+ and the packages in `requirements.txt`. The scripts run on
the CPU by default, which is how the published results were computed.

```bash
pip install -r requirements.txt
python simulation/03_main_experiment.py      # main experiment, 9,600 runs
python simulation/04_robustness.py           # robustness checks, 8,640 runs
python simulation/05_partial_rewiring.py     # partial randomisation, 4,320 runs
```

On the first run the edge list is downloaded from
<https://snap.stanford.edu/data/ego-Twitter.html> into `data/raw/` and cached as
a PyTorch tensor. Results are written to `data/results/`, overwriting the
published files. All random draws use fixed seeds, so a CPU rerun reproduces the
published results exactly. To use a GPU, set the environment variable
`NCA_DEVICE=cuda`; this is faster, but PyTorch's CUDA random number generator
produces different streams, so individual runs differ from the published files
(the results are statistically equivalent).

### 2. Analysis (R)

Requires R ≥ 4.1 and the packages `dplyr`, `tidyr`, `sandwich`, `lmtest`,
`ggplot2` and (optional) `survival`:

```r
install.packages(c("dplyr", "tidyr", "sandwich", "lmtest", "ggplot2", "survival"))
```

Open `analysis/analysis.Rproj` (or set the working directory to `analysis/`) and
run the numbered scripts in order. Tables are written to `analysis/output/` and
figures to `figures/paper/`.

### 3. Figure 1 and network diagnostics (Python)

Figure 1 is a schematic on a small generated network. It illustrates the four
seeding strategies and is not a simulation result; the script draws figures of
this kind, but the published image need not be reproduced exactly.

```bash
python figures/scripts/fig1_seeding_strategies.py   # needs numpy, matplotlib
python analysis/network_diagnostics.py              # needs numpy, scipy (optional)
```

## Notes for interpreting the results

* **Budget matching.** Mega spends the budget exactly. The other strategies add
  accounts until spend is within ±2% of the budget, skipping accounts that would
  exceed it. With the smallest budget (*K* = 1) the last gap cannot always be
  filled: 24 of 150 Micro and 17 of 150 Niche seed sets spent less than 98%
  (minimum 90.8%). ROSS divides by actual spend. `05_review_checks.R` repeats the
  strategy comparisons on the 121 of 150 replication × campaign-size
  combinations in which every strategy spent 98–102% of the budget.
* **Statistical models.** Logistic regressions of the cascade indicator on
  strategy with a fixed effect for every condition (campaign size × threshold ×
  fatigue); standard errors clustered by replication, because all strategies and
  conditions of a replication share seeds and thresholds. Conditions in which no
  run or every run cascaded are excluded from the models. Intervals describe
  Monte Carlo variation, conditional on the model and the single network used.
* **Network randomisation.** Targets of a share of the edges are shuffled, which
  keeps every account's numbers of followers and followed accounts but changes
  clustering, reciprocity and community structure at the same time. Each level
  is one randomised network; `simulation/06_rewiring_replicates.py` repeats the
  experiment on independent realisations. Self-loops and duplicate edges created
  by the shuffle are left in place and counted by `analysis/network_diagnostics.py`.
* **Follower counts are within the sample.** The most-followed account has
  3,383 followers in the network; mega- and micro-influencer tiers are relative
  to this network, not real-world follower counts.

## Data

The ego-Twitter network is part of the Stanford Large Network Dataset Collection
and is not redistributed here.

* Leskovec, J., & Krevl, A. (2014). *SNAP Datasets: Stanford large network dataset
  collection* [Data set]. <http://snap.stanford.edu/data>
* Leskovec, J., & McAuley, J. (2012). Learning to discover social circles in ego
  networks. In *Advances in Neural Information Processing Systems* (Vol. 25,
  pp. 539–547).

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

The code is released under the [MIT License](LICENSE). The SNAP data are subject
to their own terms of use.
