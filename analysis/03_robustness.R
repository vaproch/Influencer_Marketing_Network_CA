# 03_robustness.R - robustness checks (Section 3.5 of the paper)
#
#   A. Price schedule (Table 2): price exponent gamma = 0.7, 1.0 (baseline), 1.3
#   B. Network randomisation: 0%, 10%, 25%, 50%, 100% of edges randomised
#
# Both use a reduced grid with 30 replications (replications 1-30 of the main
# experiment serve as the baseline and share seeds and thresholds with the
# variants, so comparisons stay paired).
#
# Each randomisation level is ONE randomised network: replications vary seeds
# and thresholds, not the network (see simulation/06_rewiring_replicates.py).
# Randomisation preserves every account's follower and following counts but
# changes several network properties at once (clustering, reciprocity,
# communities); see analysis/network_diagnostics.py.
#
# Input : data/results/03_main_experiment.csv, 04_robustness.csv, 05_partial_rewiring.csv
# Output: analysis/output/robustness_*.csv

source(file.path("R", "helpers.R"))

# ===========================================================================
# A. Price schedule
# ===========================================================================
base <- read_results("03_main_experiment.csv") %>%
  filter(replication <= 30, fatigue_alpha %in% c(0, 0.05)) %>%
  mutate(variant = "gamma1.0")
price <- bind_rows(base, read_results("04_robustness.csv") %>%
                     filter(variant %in% c("gamma0.7", "gamma1.3"))) %>%
  mutate(variant = factor(variant, levels = c("gamma0.7", "gamma1.0", "gamma1.3")),
         # fatigue coded 0/1 here, so main effects refer to no fatigue
         alpha_u = fatigue_alpha / 0.05)

# Table 2: one model per price exponent with both interactions;
# main effects are contrasts at mean threshold 0.15 and no fatigue.
table2 <- bind_rows(lapply(levels(price$variant), function(v) {
  dv <- informative(price %>% filter(variant == v) %>% droplevels())
  tab <- fit_strategy_model(dv, ref = "Micro", moderators = c("theta_u", "alpha_u"))
  show_model(tab, paste("Price exponent", sub("gamma", "", v),
                        "- strategy x threshold + strategy x fatigue, reference Micro"))
  cbind(variant = v, tab)
}))
write_table(table2, "robustness_price_models.csv")

price_rates <- cascade_rates(price, variant, n_megas, mean_threshold_nominal,
                             fatigue_alpha, strategy) %>%
  mutate(across(c(rate, lo, hi), ~ round(.x, 3)))
write_table(price_rates, "robustness_price_rates_by_cell.csv")

seed_sets_price <- price %>%
  distinct(variant, replication, n_megas, strategy, seed_count, audience_reach,
           audience_overlap, achieved_budget, target_budget) %>%
  group_by(variant, K = n_megas, strategy) %>%
  summarise(seeds = round(mean(seed_count), 1),
            distinct_followers_reached = round(mean(audience_reach)),
            audience_overlap = round(mean(audience_overlap), 2),
            spend_share_min = round(min(achieved_budget / target_budget), 3),
            .groups = "drop")
write_table(seed_sets_price, "robustness_price_seed_sets.csv")

ross_price <- price %>%
  filter(fatigue_alpha == 0) %>%
  group_by(variant, n_megas, mean_threshold_nominal, strategy) %>%
  summarise(runs = n(), mean_ross = round(mean(ross), 2), .groups = "drop")
write_table(ross_price, "robustness_price_ross_no_fatigue.csv")

# ===========================================================================
# B. Network randomisation (no fatigue)
# ===========================================================================
rw <- read_rewiring_series() %>%
  mutate(cond = interaction(K, theta_f, drop = TRUE))

cat("\nAudience overlap of the seed sets (K = 3) by share of randomised edges:\n")
print(rw %>% filter(n_megas == 3) %>%
        distinct(pct, replication, strategy, audience_overlap) %>%
        group_by(pct, strategy) %>%
        summarise(overlap = round(mean(audience_overlap), 2), .groups = "drop") %>%
        pivot_wider(names_from = strategy, values_from = overlap))

rw_rates <- cascade_rates(rw, pct, n_megas, mean_threshold_nominal, strategy) %>%
  mutate(across(c(rate, lo, hi), ~ round(.x, 3)))
write_table(rw_rates, "robustness_rewiring_rates_by_cell.csv")

cat("\nCascade rates pooled over campaign sizes (90 runs per cell):\n")
rw_pooled <- cascade_rates(rw, pct, mean_threshold_nominal, strategy) %>%
  mutate(rate = round(rate, 3))
print(rw_pooled %>% select(pct, mean_threshold_nominal, strategy, rate) %>%
        pivot_wider(names_from = strategy, values_from = rate), n = Inf)
write_table(rw_pooled, "robustness_rewiring_rates_pooled_over_K.csv")

cat("\nFully randomised network: largest cascade counts in any single cell (30 runs):\n")
print(rw_rates %>% filter(pct == 100) %>% arrange(desc(cascades)) %>% head(4))

rw_models <- bind_rows(lapply(c(0, 10, 25, 50, 100), function(p) {
  dp <- informative(rw %>% filter(pct == p) %>% droplevels())
  if (nlevels(factor(dp$cond)) < 2) {
    cat(sprintf("\n%d%% randomised: fewer than two informative conditions; no model.\n", p))
    return(NULL)
  }
  a <- fit_strategy_model(dp, ref = "Micro")
  b <- fit_strategy_model(dp, ref = "Micro", moderators = "theta_u")
  b <- b[grepl("threshold", b$term), ]
  attr(b, "info") <- attr(a, "info")
  show_model(a, sprintf("%d%% of edges randomised - pooled contrasts, reference Micro", p))
  show_model(b, sprintf("%d%% of edges randomised - strategy x threshold", p))
  cbind(pct_randomised = p, bind_rows(a, b),
        conditions = attr(a, "info")[["conditions"]])
}))
write_table(rw_models, "robustness_rewiring_models.csv")
