rm(list = ls())
graphics.off()
cat("\014")
library(dplyr)
library(tidyr)

df <- read.csv("simulation_results_v2.csv")

df <- df %>% mutate(
  strategy    = factor(strategy, levels = c("Mega", "Micro", "Random")),
  replication = factor(replication),
  K           = factor(n_megas),
  alpha_f     = factor(fatigue_alpha),
  theta_f     = factor(mean_threshold_nominal),
  induced     = final_adopters - seed_count,
  cascade     = market_saturation >= 0.10    # provisional cutoff, checked below
)

# --- 1. Design check: expect TRUE (50 runs per cell) ---
all(table(df$strategy, df$alpha_f, df$theta_f, df$K) == 50)

# --- 2. Budget matching and seed counts by campaign size ---
df %>%
  mutate(budget_ratio = achieved_budget / target_budget) %>%
  group_by(K, strategy) %>%
  summarise(seeds = mean(seed_count), budget = mean(target_budget),
            ratio_min = min(budget_ratio), ratio_mean = mean(budget_ratio),
            ratio_max = max(budget_ratio), .groups = "drop") %>%
  print(n = Inf)

# --- 3. Is the cascade cutoff still sensible? Count runs by saturation band ---
table(cut(df$market_saturation,
          breaks = c(0, 0.001, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 1),
          include.lowest = TRUE))

# --- 4. Main table: cascade rate (share of 50 replications) ---
# rows = campaign size K, threshold, strategy; columns = fatigue alpha
df %>%
  group_by(K, theta_f, strategy, alpha_f) %>%
  summarise(rate = mean(cascade), .groups = "drop") %>%
  pivot_wider(names_from = alpha_f, values_from = rate) %>%
  print(n = Inf)

# --- 5. Overall cascade rate by strategy and campaign size ---
df %>%
  group_by(K, strategy) %>%
  summarise(rate = mean(cascade), .groups = "drop") %>%
  pivot_wider(names_from = strategy, values_from = rate)



library(survival)

# --- 1. Do conclusions depend on the cascade cutoff? ---
for (cut_off in c(0.10, 0.25, 0.50)) {
  cat("\nCascade = saturation >=", cut_off, "\n")
  print(df %>% group_by(K, strategy) %>%
          summarise(rate = mean(market_saturation >= cut_off), .groups = "drop") %>%
          pivot_wider(names_from = strategy, values_from = rate))
}

# --- 2. Where do the partial cascades occur? ---
band <- cut(df$market_saturation, c(0, 0.10, 0.60, 1), include.lowest = TRUE,
            labels = c("under10", "10to60", "over60"))
table(band, df$theta_f)
table(band, df$alpha_f)

# --- 0. Packages (installs them once if missing) ---
for (p in c("sandwich", "lmtest"))
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
library(sandwich)
library(lmtest)

# --- 1. Prepare variables ---
df <- df %>% mutate(
  cond       = interaction(K, theta_f, alpha_f, drop = TRUE),   # one of 60 conditions
  theta_u    = (mean_threshold_nominal - 0.15) / 0.05,          # units of 0.05
  alpha_u    = (fatigue_alpha - 0.05) / 0.05,                   # units of 0.05
  logK_c     = log10(n_megas) - 0.5,                            # per tenfold campaign size
  strategy_r = relevel(strategy, ref = "Random")
)

# Keep only conditions where a cascade both happened and did not happen
# (conditions with all-or-nothing outcomes carry no information about strategy)
informative <- df %>% group_by(cond) %>%
  summarise(ok = n_distinct(cascade) == 2, .groups = "drop") %>%
  filter(ok) %>% pull(cond)
d <- df %>% filter(cond %in% informative) %>% droplevels()
nrow(d); nlevels(d$cond)

# --- 2. Helper: odds ratios with cluster-robust confidence intervals ---
or_table <- function(m) {
  ct <- coeftest(m, vcov = vcovCL(m, cluster = d$replication))
  out <- data.frame(term = rownames(ct), OR = exp(ct[, 1]),
                    lo = exp(ct[, 1] - 1.96 * ct[, 2]),
                    hi = exp(ct[, 1] + 1.96 * ct[, 2]), p = ct[, 4])
  out <- out[grepl("strategy", out$term), ]
  out[, -1] <- round(out[, -1], 3)
  out
}

# --- 3. Models ---
# 3a. Overall: Micro and Random versus Mega
m1  <- glm(cascade ~ strategy + cond, family = binomial, data = d)
or_table(m1)

# 3b. Micro versus Random
m1b <- glm(cascade ~ strategy_r + cond, family = binomial, data = d)
or_table(m1b)

# 3c. Does the strategy effect change with threshold, fatigue, campaign size?
m2 <- glm(cascade ~ strategy + strategy:theta_u + cond, family = binomial, data = d)
or_table(m2)
m3 <- glm(cascade ~ strategy + strategy:alpha_u + cond, family = binomial, data = d)
or_table(m3)
m4 <- glm(cascade ~ strategy + strategy:logK_c + cond, family = binomial, data = d)
or_table(m4)


if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
library(ggplot2)

# --- 1. Expected ROSS (mean over all runs, failed campaigns included) ---
df %>% group_by(K, strategy) %>%
  summarise(mean_ross = mean(ross), mean_induced = mean(induced), .groups = "drop") %>%
  print()

df %>% group_by(K, theta_f, strategy) %>%
  summarise(mean_ross = mean(ross), .groups = "drop") %>%
  pivot_wider(names_from = strategy, values_from = mean_ross) %>%
  mutate(micro_over_mega = Micro / Mega, random_over_mega = Random / Mega) %>%
  print(n = Inf)

# --- 2. Figure 1: cascade probability by threshold, fatigue, campaign size ---
a_lv <- paste0("alpha = ", sort(unique(df$fatigue_alpha)))
f1 <- df %>%
  mutate(K_lab = factor(paste0("K = ", K), levels = paste0("K = ", c(1, 3, 10))),
         a_lab = factor(paste0("alpha = ", fatigue_alpha), levels = a_lv)) %>%
  group_by(K_lab, a_lab, theta = mean_threshold_nominal, strategy) %>%
  summarise(p = mean(cascade), n = n(), .groups = "drop") %>%
  mutate(se = sqrt(p * (1 - p) / n), lo = pmax(0, p - 1.96 * se), hi = pmin(1, p + 1.96 * se))

fig1 <- ggplot(f1, aes(theta, p, colour = strategy, shape = strategy)) +
  geom_ribbon(aes(ymin = lo, ymax = hi, fill = strategy), alpha = 0.12, colour = NA) +
  geom_line() + geom_point(size = 1.8) +
  facet_grid(a_lab ~ K_lab) +
  labs(x = "Mean adoption threshold", y = "Probability of a global cascade",
       colour = "Strategy", shape = "Strategy", fill = "Strategy") +
  theme_bw()
ggsave("fig1_cascade_probability.png", fig1, width = 10, height = 8, dpi = 300)

# --- 3. Figure 2: expected ROSS by campaign size ---
f2 <- df %>% group_by(K, strategy) %>%
  summarise(m = mean(ross), se = sd(ross) / sqrt(n()), .groups = "drop")

fig2 <- ggplot(f2, aes(K, m, colour = strategy, group = strategy)) +
  geom_line() +
  geom_pointrange(aes(ymin = m - 1.96 * se, ymax = m + 1.96 * se),
                  position = position_dodge(width = 0.2)) +
  labs(x = "Campaign size K (number of mega-influencers the budget buys)",
       y = "Expected ROSS (induced adopters per unit of spend)", colour = "Strategy") +
  theme_bw()
ggsave("fig2_expected_ross.png", fig2, width = 7, height = 5, dpi = 300)

getwd()   # the two PNG files are saved in this folder
