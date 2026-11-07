# 01_descriptives.R - descriptive results of the main experiment
# (Sections 2.3 and 3.1 of the paper, and supporting tables).
#
# Input : data/results/03_main_experiment.csv
# Output: analysis/output/*.csv (tables listed at the end of each block)
#
# Unless a table says otherwise, a rate pools every condition that is not
# shown in it. Counts (cascades / runs) are reported next to every rate.

source(file.path("R", "helpers.R"))
d <- read_main()

# --- Design check: 4 strategies x 3 campaign sizes x 4 thresholds x 4 fatigue
#     levels x 50 replications = 9,600 runs -----------------------------------
stopifnot(nrow(d) == 9600,
          all(table(d$strategy, d$K, d$theta_f, d$alpha_f) == 50))
cat("Design check passed: 9,600 runs, 50 per cell.\n")

# --- Realised mean thresholds (truncation raises them at low nominal values) --
realised <- d %>%
  group_by(mean_threshold_nominal) %>%
  summarise(realised_mean_threshold = round(mean(mean_threshold_realized), 3))
print(realised)
write_table(realised, "realised_thresholds.csv")

# --- Seed sets bought at equal budget (one row per replication x K x strategy)
seed_sets <- d %>%
  distinct(replication, n_megas, strategy, seed_count, target_budget,
           achieved_budget, audience_reach, audience_overlap) %>%
  group_by(K = n_megas, strategy) %>%
  summarise(seed_sets = n(),
            budget = round(mean(target_budget)),
            seeds = round(mean(seed_count), 1),
            distinct_followers_reached = round(mean(audience_reach)),
            audience_overlap = round(mean(audience_overlap), 2),
            spend_share_mean = round(mean(achieved_budget / target_budget), 3),
            spend_share_min = round(min(achieved_budget / target_budget), 3),
            .groups = "drop")
cat("\nSeed sets at equal budget (means over 50 replications):\n")
print(as.data.frame(seed_sets), row.names = FALSE)
write_table(seed_sets, "seed_sets.csv")

# --- Bimodality of campaign outcomes -------------------------------------------
bands <- d %>%
  mutate(band = cut(market_saturation, c(0, 0.05, 0.10, 0.60, 1),
                    labels = c("< 5%", "5-10%", "10-60%", "> 60%"),
                    include.lowest = TRUE)) %>%
  count(band, name = "runs") %>%
  mutate(share = round(runs / sum(runs), 3))
cat("\nFinal adoption share, all 9,600 runs:\n")
print(bands)
write_table(bands, "outcome_bands.csv")

size <- d %>% filter(cascade) %>%
  summarise(cascades = n(),
            median = median(market_saturation),
            q25 = quantile(market_saturation, 0.25),
            q75 = quantile(market_saturation, 0.75))
cat("\nSize of global cascades (final adoption share):\n")
print(round(size, 4))

# --- Cutoff sensitivity: ordering of strategies under 10%, 25%, 50% cutoffs ----
# (pooled over thresholds and fatigue levels; 800 runs per K x strategy)
cutoffs <- bind_rows(lapply(c(0.10, 0.25, 0.50), function(cut) {
  d %>%
    group_by(cutoff = cut, K, strategy) %>%
    summarise(cascades = sum(market_saturation >= cut), runs = n(), .groups = "drop") %>%
    mutate(rate = round(cascades / runs, 3))
}))
cat("\nCascade rates under different cutoffs (pooled over thresholds and fatigue):\n")
print(cutoffs %>% select(cutoff, K, strategy, rate) %>%
        pivot_wider(names_from = strategy, values_from = rate))
write_table(cutoffs, "cutoff_sensitivity.csv")

# --- Cascade rates in every cell (50 runs each), with Wilson 95% intervals ----
cells <- cascade_rates(d, n_megas, mean_threshold_nominal, fatigue_alpha, strategy) %>%
  mutate(across(c(rate, lo, hi), ~ round(.x, 3)))
write_table(cells, "cascade_rates_by_cell.csv")
cat("\nCascade rates by cell written to output/cascade_rates_by_cell.csv\n")

cat("\nCascade rates by threshold (pooled over K and fatigue; 600 runs per cell):\n")
print(cascade_rates(d, mean_threshold_nominal, strategy) %>%
        mutate(rate = round(rate, 3)) %>%
        select(mean_threshold_nominal, strategy, rate) %>%
        pivot_wider(names_from = strategy, values_from = rate))

# --- Speed of cascades: rounds until 10% adoption (cascades with mu <= 0.15) ---
speed <- d %>%
  filter(cascade, mean_threshold_nominal <= 0.15) %>%
  mutate(ticks_to_10pct = as.numeric(ticks_to_10pct)) %>%
  group_by(fatigue_alpha, strategy) %>%
  summarise(cascades = n(), median_rounds_to_10pct = median(ticks_to_10pct),
            .groups = "drop")
cat("\nRounds until 10% adoption (successful campaigns, mean threshold <= 0.15):\n")
print(speed %>% select(-cascades) %>%
        pivot_wider(names_from = strategy, values_from = median_rounds_to_10pct))
write_table(speed, "rounds_to_10pct.csv")

# --- Return on seed spend (ROSS) -----------------------------------------------
# ROSS = (final adopters - seeds) / actual spend, in simulated additional
# adopters per cost unit of the model's price scale (followers ^ gamma).
ross_cell <- d %>%
  group_by(n_megas, mean_threshold_nominal, fatigue_alpha, strategy) %>%
  summarise(runs = n(), mean_ross = mean(ross), .groups = "drop")
ross_pooled <- d %>%
  group_by(n_megas, mean_threshold_nominal, strategy) %>%
  summarise(runs = n(), mean_ross = mean(ross), .groups = "drop")
cat("\nMean ROSS by campaign size and threshold (pooled over the 4 fatigue levels):\n")
print(ross_pooled %>% mutate(mean_ross = round(mean_ross, 2)) %>% select(-runs) %>%
        pivot_wider(names_from = strategy, values_from = mean_ross), n = Inf)
write_table(ross_cell, "ross_by_cell.csv")
write_table(ross_pooled, "ross_pooled_over_fatigue.csv")
