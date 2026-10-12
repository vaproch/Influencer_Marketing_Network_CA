rm(list=ls())
graphics.off()
cat("\014")

df <- read.csv("simulation_results.csv")

library(dplyr)

# --- 1. Types and labels ---
df <- df %>%
  mutate(
    strategy      = factor(strategy, levels = c("Mega", "Micro", "Random")),
    replication   = factor(replication),
    alpha_f       = factor(fatigue_alpha),
    theta_f       = factor(mean_threshold_nominal)
  )

# --- 2. Design completeness: expect 30 per cell, 1800 rows total ---
nrow(df)
anyNA(df)
table(df$strategy, df$alpha_f, df$theta_f)[, , 1]   # one theta slice is enough
all(table(df$strategy, df$alpha_f, df$theta_f) == 30)

# --- 3. Budget matching: achieved / target spend by strategy ---
df %>%
  mutate(budget_ratio = achieved_budget / target_budget) %>%
  group_by(strategy) %>%
  summarise(min = min(budget_ratio), mean = mean(budget_ratio), max = max(budget_ratio))

# --- 4. Seeds and budget scale ---
df %>%
  group_by(strategy) %>%
  summarise(seeds_mean = mean(seed_count), seeds_min = min(seed_count), seeds_max = max(seed_count),
            budget_mean = mean(target_budget), budget_min = min(target_budget), budget_max = max(target_budget))

# --- 5. Realized vs nominal threshold ---
df %>%
  group_by(mean_threshold_nominal) %>%
  summarise(realized = mean(mean_threshold_realized))

# --- 6. Outcome sanity ---
summary(df[, c("convergence_tick", "exposed_nodes", "final_adopters", "market_saturation", "ross")])


# --- 1. Shape of the outcome ---
hist(log10(df$market_saturation), breaks = 60,
     main = "Market saturation (log10)", xlab = "log10(final adopters / N)")

table(cut(df$market_saturation,
          breaks = c(0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1),
          include.lowest = TRUE))

# --- 2. Share of runs that never spread beyond the seeds ---
mean(df$convergence_tick == 0)
df %>% group_by(strategy) %>% summarise(no_spread = mean(convergence_tick == 0))

# --- 3. Where do the big outcomes occur? (cutoff 10% is provisional) ---
df <- df %>% mutate(cascade = market_saturation >= 0.10)

df %>% group_by(strategy, theta_f) %>%
  summarise(cascade_rate = mean(cascade), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = theta_f, values_from = cascade_rate)

df %>% group_by(strategy, alpha_f) %>%
  summarise(cascade_rate = mean(cascade), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = alpha_f, values_from = cascade_rate)

# --- 4. Medians and means by strategy x theta (all alphas pooled) ---
df %>% group_by(strategy, theta_f) %>%
  summarise(median_sat = median(market_saturation),
            mean_sat   = mean(market_saturation),
            median_ross = median(ross), mean_ross = mean(ross),
            .groups = "drop")


df <- df %>% mutate(induced = final_adopters - seed_count)

# 1. Cascades out of 30 replications, in the 6 live cells
df %>% filter(mean_threshold_nominal <= 0.2, fatigue_alpha <= 0.1) %>%
  group_by(theta_f, alpha_f, strategy) %>%
  summarise(cascades = sum(cascade), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = strategy, values_from = cascades)

# 2. Do high-threshold cells diffuse at all? (induced adopters, not saturation)
df %>% filter(mean_threshold_nominal >= 0.3) %>%
  group_by(strategy, theta_f) %>%
  summarise(median_induced = median(induced), p95_induced = quantile(induced, .95),
            max_induced = max(induced), .groups = "drop")

# 3. Budget confound at theta = 0.1: does a bigger mega budget mean more cascades?
df %>% filter(mean_threshold_nominal == 0.1, fatigue_alpha <= 0.1) %>%
  mutate(budget_bin = cut(target_budget,
                          quantile(df$target_budget, c(0, 1/3, 2/3, 1)),
                          include.lowest = TRUE)) %>%
  group_by(budget_bin, strategy) %>%
  summarise(n = n(), cascade_rate = mean(cascade), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = strategy, values_from = c(cascade_rate, n))

# 4. Given a cascade: size, speed, efficiency by strategy
df %>% filter(cascade) %>%
  group_by(strategy) %>%
  summarise(n = n(), median_sat = median(market_saturation),
            median_tick = median(convergence_tick), median_ross = median(ross))
