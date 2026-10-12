rm(list=ls())
graphics.off()
cat("\014")

library(dplyr)
library(tidyr)
for (p in c("sandwich", "lmtest"))
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
library(sandwich)
library(lmtest)

v2 <- read.csv("simulation_results_v2.csv")
v3 <- read.csv("simulation_results_v3.csv")

# --- 1. Reproduction check: Mega/Micro/Random in v3 must equal v2 ---
key <- c("replication", "strategy", "n_megas", "fatigue_alpha", "mean_threshold_nominal")
chk <- inner_join(v2[, c(key, "final_adopters", "seed_count")],
                  v3[v3$strategy != "Niche", c(key, "final_adopters", "seed_count")],
                  by = key, suffix = c("_v2", "_v3"))
nrow(chk)                                        # expect 7200
mean(chk$final_adopters_v2 == chk$final_adopters_v3)   # expect 1
mean(chk$seed_count_v2 == chk$seed_count_v3)           # expect 1

# --- 2. Prepare v3 ---
v3 <- v3 %>% mutate(
  strategy = factor(strategy, levels = c("Mega", "Micro", "Niche", "Random")),
  K        = factor(n_megas),
  alpha_f  = factor(fatigue_alpha),
  theta_f  = factor(mean_threshold_nominal),
  induced  = final_adopters - seed_count,
  cascade  = market_saturation >= 0.10
)
all(table(v3$strategy, v3$alpha_f, v3$theta_f, v3$K) == 50)   # expect TRUE

# --- 3. Seeds, budget matching, audience overlap ---
v3 %>%
  mutate(budget_ratio = achieved_budget / target_budget) %>%
  group_by(K, strategy) %>%
  summarise(seeds = mean(seed_count), ratio_min = min(budget_ratio),
            ratio_mean = mean(budget_ratio), reach = mean(audience_reach),
            overlap = mean(audience_overlap), .groups = "drop") %>%
  print(n = Inf)

# --- 4. Cascade rate: rows = K, theta, alpha; columns = strategy ---
v3 %>%
  group_by(K, theta_f, alpha_f, strategy) %>%
  summarise(rate = mean(cascade), .groups = "drop") %>%
  pivot_wider(names_from = strategy, values_from = rate) %>%
  print(n = Inf)

# --- 5. Paired models (same logic as before) ---
v3 <- v3 %>% mutate(
  cond    = interaction(K, theta_f, alpha_f, drop = TRUE),
  theta_u = (mean_threshold_nominal - 0.15) / 0.05,
  alpha_u = (fatigue_alpha - 0.05) / 0.05
)
informative <- v3 %>% group_by(cond) %>%
  summarise(ok = n_distinct(cascade) == 2, .groups = "drop") %>%
  filter(ok) %>% pull(cond)
d <- v3 %>% filter(cond %in% informative) %>% droplevels()
nrow(d); nlevels(d$cond)

or_table <- function(m) {
  ct <- coeftest(m, vcov = vcovCL(m, cluster = d$replication))
  out <- data.frame(term = rownames(ct), OR = exp(ct[, 1]),
                    lo = exp(ct[, 1] - 1.96 * ct[, 2]),
                    hi = exp(ct[, 1] + 1.96 * ct[, 2]), p = ct[, 4])
  out <- out[grepl("strategy", out$term), ]
  out[, -1] <- round(out[, -1], 3)
  rownames(out) <- NULL
  out
}

d$s_mega   <- relevel(d$strategy, ref = "Mega")
d$s_micro  <- relevel(d$strategy, ref = "Micro")
d$s_random <- relevel(d$strategy, ref = "Random")

cat("\nReference = Mega\n");   or_table(glm(cascade ~ s_mega   + cond, family = binomial, data = d))
cat("\nReference = Micro\n");  or_table(glm(cascade ~ s_micro  + cond, family = binomial, data = d))
cat("\nReference = Random\n"); or_table(glm(cascade ~ s_random + cond, family = binomial, data = d))

cat("\nStrategy x threshold (reference = Micro)\n")
or_table(glm(cascade ~ s_micro + s_micro:theta_u + cond, family = binomial, data = d))
cat("\nStrategy x fatigue (reference = Micro)\n")
or_table(glm(cascade ~ s_micro + s_micro:alpha_u + cond, family = binomial, data = d))

# --- 6. Speed: ticks to reach 10% adoption (cascades only, easy thresholds) ---
v3 %>% filter(cascade, mean_threshold_nominal <= 0.15) %>%
  group_by(alpha_f, strategy) %>%
  summarise(n = n(), median_ticks_to_10pct = median(ticks_to_10pct), .groups = "drop") %>%
  print(n = Inf)

library(ggplot2)

# --- 1. Corrected odds-ratio table ---
or_table <- function(m) {
  ct <- coeftest(m, vcov = vcovCL(m, cluster = d$replication))
  out <- data.frame(term = rownames(ct), OR = exp(ct[, 1]),
                    lo = exp(ct[, 1] - 1.96 * ct[, 2]),
                    hi = exp(ct[, 1] + 1.96 * ct[, 2]), p = ct[, 4])
  out <- out[grepl("^s_", out$term), ]
  out[, -1] <- round(out[, -1], 3)
  rownames(out) <- NULL
  out
}

# --- 2. Overall strategy comparisons (odds of a global cascade) ---
cat("\nReference = Micro\n");  or_table(glm(cascade ~ s_micro  + cond, family = binomial, data = d))
cat("\nReference = Mega\n");   or_table(glm(cascade ~ s_mega   + cond, family = binomial, data = d))
cat("\nReference = Random\n"); or_table(glm(cascade ~ s_random + cond, family = binomial, data = d))

# --- 3. Does the strategy gap change with threshold and fatigue? (reference = Micro) ---
cat("\nStrategy x threshold\n")
or_table(glm(cascade ~ s_micro * theta_u + cond, family = binomial, data = d))
cat("\nStrategy x fatigue\n")
or_table(glm(cascade ~ s_micro * alpha_u + cond, family = binomial, data = d))

# --- 4. Pooled cascade rates ---
v3 %>% group_by(theta_f, strategy) %>% summarise(rate = mean(cascade), .groups = "drop") %>%
  pivot_wider(names_from = strategy, values_from = rate)
v3 %>% group_by(K, theta_f, strategy) %>% summarise(rate = mean(cascade), .groups = "drop") %>%
  pivot_wider(names_from = strategy, values_from = rate) %>% print(n = Inf)

# --- 5. Mean ROSS by threshold (cell-wise, not pooled across thresholds) ---
v3 %>% group_by(K, theta_f, strategy) %>% summarise(ross = mean(ross), .groups = "drop") %>%
  pivot_wider(names_from = strategy, values_from = ross) %>% print(n = Inf)

# --- 6. Figure: cascade probability, four strategies ---
f1 <- v3 %>%
  mutate(K_lab = factor(paste0("K = ", K), levels = paste0("K = ", c(1, 3, 10))),
         a_lab = factor(paste0("alpha = ", fatigue_alpha))) %>%
  group_by(K_lab, a_lab, theta = mean_threshold_nominal, strategy) %>%
  summarise(p = mean(cascade), n = n(), .groups = "drop") %>%
  mutate(se = sqrt(p * (1 - p) / n), lo = pmax(0, p - 1.96 * se), hi = pmin(1, p + 1.96 * se))

fig <- ggplot(f1, aes(theta, p, colour = strategy, shape = strategy)) +
  geom_ribbon(aes(ymin = lo, ymax = hi, fill = strategy), alpha = 0.10, colour = NA) +
  geom_line() + geom_point(size = 1.8) +
  facet_grid(a_lab ~ K_lab) +
  labs(x = "Mean adoption threshold", y = "Probability of a global cascade",
       colour = "Strategy", shape = "Strategy", fill = "Strategy") +
  theme_bw()
ggsave("fig3_cascade_probability_v3.png", fig, width = 10, height = 8, dpi = 300)
