# Simulation results

One row per simulation run. Each file is written by the script with the same
number in [`simulation/`](../../simulation).

| File | Script | Runs | Design | Used in the paper |
|---|---|---|---|---|
| `01_pilot.csv` | `01_pilot.py` | 1,800 | 3 strategies (Mega, Micro, Random) × 4 thresholds (0.10–0.40) × 5 fatigue rates (0–0.40) × 30 replications; budget = one mega-influencer | Pilot (Section 2.4) |
| `02_extended_grid.csv` | `02_extended_grid.py` | 9,000 | 3 strategies × 3 campaign sizes × 5 thresholds (0.10–0.30) × 4 fatigue rates (0–0.10) × 50 replications | Design development only |
| `03_main_experiment.csv` | `03_main_experiment.py` | 9,600 | 4 strategies × 3 campaign sizes (1, 3, 10) × 4 thresholds (0.10–0.25) × 4 fatigue rates (0, 0.02, 0.05, 0.10) × 50 replications | **Main results** (Sections 3.1–3.4) |
| `04_robustness.csv` | `04_robustness.py` | 8,640 | 3 variants × 4 strategies × 3 campaign sizes × 4 thresholds × 2 fatigue rates (0, 0.05) × 30 replications | Robustness (Section 3.5) |
| `05_partial_rewiring.csv` | `05_partial_rewiring.py` | 4,320 | 3 randomisation levels × 4 strategies × 3 campaign sizes × 4 thresholds × no fatigue × 30 replications | Robustness (Section 3.5) |

Replication *r* of every file uses the same random seed (`12345 + r`). Within a
replication, all strategies and conditions share the consumer thresholds, and
the Mega, Micro and Random seed sets of stages 2–5 are identical wherever the
prices are the same (all variants except the two price exponents). Comparisons
between strategies, and between the baseline and the robustness variants, are
therefore paired. Each randomisation level uses a single randomised network.

## Columns

| Column | Meaning |
|---|---|
| `variant` | (files 04, 05 only) `gamma0.7` / `gamma1.3`: price exponent γ; `rewired`: 100% of edges randomised; `rewire10` / `rewire25` / `rewire50`: 10 / 25 / 50% of edges randomised. Rows without this column use the original network and γ = 1. |
| `run_id` | Running number of the run within the file |
| `replication` | Replication index; determines all random draws (seed sets, thresholds) |
| `strategy` | Seeding strategy: `Mega`, `Micro`, `Niche`, `Random` |
| `n_megas` | Campaign size *K*: the budget equals the price of *K* mega-influencers (missing in 01: *K* = 1) |
| `fatigue_alpha` | Fatigue rate α. Receptiveness is exp(−α · E), where E counts earlier rounds in which the consumer was exposed (at least one followed account had adopted) but did not adopt. It grows by one per exposed round, regardless of how many followed accounts are active. |
| `mean_threshold_nominal` | Mean μ of the adoption-threshold distribution (normal, SD 0.10, truncated to [0.05, 0.95]) |
| `mean_threshold_realized` | Realised mean threshold across all accounts in the run |
| `seed_count` | Number of seed accounts bought |
| `target_budget` | Budget B: total price of the *K* mega-influencers (price = followers^γ) |
| `achieved_budget` | Amount actually spent by this strategy. Mega spends exactly B; the other strategies stop once spend is within ±2% of B and skip accounts that would exceed B, so they can end below 98% when the last gap cannot be filled (lowest observed: 90.8%). |
| `audience_reach` | Number of distinct followers of the seed accounts |
| `audience_overlap` | Total followers of the seeds divided by `audience_reach` (1 = no overlap) |
| `convergence_tick` | Number of rounds until no further adoption occurred |
| `ticks_to_10pct` | First round in which at least 10% of accounts had adopted (`NA` if never) |
| `exposed_nodes` | Accounts exposed to the campaign at least once |
| `final_adopters` | Adopters at the end of the run, including the seeds |
| `market_saturation` | `final_adopters` divided by the number of accounts (81,306) |
| `ross` | Return on seed spend: (`final_adopters` − `seed_count`) / `achieved_budget`, i.e. simulated additional adopters per cost unit of the model's price scale. It is not a monetary return and is not comparable across price exponents. |

A run is counted as a **global cascade** when `market_saturation` ≥ 0.10.
