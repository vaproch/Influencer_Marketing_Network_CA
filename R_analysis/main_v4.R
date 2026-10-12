rm(list=ls())
graphics.off()
cat("\014")

library(dplyr)
library(tidyr)
library(sandwich)
library(lmtest)

v3 <- read.csv("simulation_results_v3.csv")
v4 <- read.csv("simulation_results_v4_robustness.csv")

# --- 1. Pairing check: rewired Mega/Micro/Random seeds must equal v3 ---
key <- c("replication", "strategy", "n_megas", "fatigue_alpha", "mean_threshold_nominal")
a <- v4 %>% filter(variant == "rewired", strategy != "Niche") %>%
  select(all_of(key), seed_count, achieved_budget)
b <- v3 %>% filter(strategy != "Niche") %>%
  select(all_of(key), seed_count, achieved_budget)
chk <- inner_join(a, b, by = key, suffix = c("_rw", "_v3"))
nrow(chk)                                              # expect 2160
mean(chk$seed_count_rw == chk$seed_count_v3)           # expect 1
mean(chk$achieved_budget_rw == chk$achieved_budget_v3) # expect 1

# --- 2. Combine: original network (from v3, same grid) + the 3 variants ---
base <- v3 %>% filter(replication <= 30, fatigue_alpha %in% c(0, 0.05)) %>%
  mutate(variant = "base")

dat <- bind_rows(base, v4) %>% mutate(
  variant  = factor(variant, levels = c("base", "rewired", "gamma0.7", "gamma1.3")),
  strategy = factor(strategy, levels = c("Mega", "Micro", "Niche", "Random")),
  K        = factor(n_megas),
  alpha_f  = factor(fatigue_alpha),
  theta_f  = factor(mean_threshold_nominal),
  cascade  = market_saturation >= 0.10
)
nrow(dat)                                                                  # expect 11520
all(table(dat$variant, dat$strategy, dat$theta_f, dat$alpha_f, dat$K) == 30)  # expect TRUE

# --- 3. Budget matching, and how much the Niche seeds overlap ---
dat %>% mutate(ratio = achieved_budget / target_budget) %>%
  group_by(variant, strategy) %>%
  summarise(ratio_min = min(ratio), ratio_mean = mean(ratio), .groups = "drop") %>%
  print(n = Inf)

dat %>% filter(K == "3") %>%
  group_by(variant, strategy) %>%
  summarise(seeds = mean(seed_count), reach = mean(audience_reach),
            overlap = mean(audience_overlap), .groups = "drop") %>%
  print(n = Inf)

# --- 4. Cascade rate (pooled over campaign sizes K) ---
dat %>% group_by(variant, theta_f, alpha_f, strategy) %>%
  summarise(rate = mean(cascade), .groups = "drop") %>%
  pivot_wider(names_from = strategy, values_from = rate) %>%
  print(n = Inf)

# --- 5. Odds of a cascade relative to Micro, separately at each threshold ---
or_slice <- function(s) {
  s <- s %>% mutate(cond = interaction(K, alpha_f, drop = TRUE))
  info <- s %>% group_by(cond) %>%
    summarise(ok = n_distinct(cascade) == 2, .groups = "drop") %>%
    filter(ok) %>% pull(cond)
  d <- s %>% filter(cond %in% info) %>% droplevels()
  if (nrow(d) == 0) return(NULL)
  d$s <- relevel(d$strategy, ref = "Micro")
  f <- if (nlevels(d$cond) > 1) cascade ~ s + cond else cascade ~ s
  m <- tryCatch(suppressWarnings(glm(f, family = binomial, data = d)), error = function(e) NULL)
  if (is.null(m)) return(NULL)
  ct <- tryCatch(coeftest(m, vcov = vcovCL(m, cluster = d$replication)), error = function(e) NULL)
  if (is.null(ct)) return(NULL)
  out <- data.frame(term = rownames(ct), OR = exp(ct[, 1]),
                    lo = exp(ct[, 1] - 1.96 * ct[, 2]),
                    hi = exp(ct[, 1] + 1.96 * ct[, 2]), p = ct[, 4],
                    n_conditions = nlevels(d$cond))
  out[grepl("^s", out$term), ]
}

res <- list()
for (v in levels(dat$variant)) for (th in levels(dat$theta_f)) {
  r <- or_slice(dat %>% filter(variant == v, theta_f == th))
  if (!is.null(r)) res[[paste(v, th)]] <- cbind(variant = v, theta = th, r)
}
tab <- bind_rows(res)
tab[, c("OR", "lo", "hi")] <- round(tab[, c("OR", "lo", "hi")], 2)
tab$p <- round(tab$p, 3)
rownames(tab) <- NULL
print(tab, row.names = FALSE)

# --- 6. Interaction models per variant (reference = Micro) ---
# theta_u: per +0.05 of threshold (0 = 0.15); alpha_u: 0 -> 0.05 of fatigue
for (v in levels(dat$variant)) {
  s <- dat %>% filter(variant == v) %>%
    mutate(cond = interaction(K, theta_f, alpha_f, drop = TRUE),
           theta_u = (mean_threshold_nominal - 0.15) / 0.05,
           alpha_u = fatigue_alpha / 0.05)
  info <- s %>% group_by(cond) %>%
    summarise(ok = n_distinct(cascade) == 2, .groups = "drop") %>%
    filter(ok) %>% pull(cond)
  d <- s %>% filter(cond %in% info) %>% droplevels()
  d$s <- relevel(d$strategy, ref = "Micro")
  cat("\n=== Variant:", v, "| informative conditions:", nlevels(d$cond), "===\n")
  out <- tryCatch({
    m  <- suppressWarnings(glm(cascade ~ s * theta_u + s * alpha_u + cond,
                               family = binomial, data = d))
    ct <- coeftest(m, vcov = vcovCL(m, cluster = d$replication))
    o  <- data.frame(term = rownames(ct), OR = exp(ct[, 1]),
                     lo = exp(ct[, 1] - 1.96 * ct[, 2]),
                     hi = exp(ct[, 1] + 1.96 * ct[, 2]), p = ct[, 4])
    o <- o[grepl("^s", o$term), ]
    o[, -1] <- round(o[, -1], 3)
    rownames(o) <- NULL
    o
  }, error = function(e) paste("model failed:", conditionMessage(e)))
  print(out)
}

# --- 7. Efficiency (ROSS), single small campaign, no fatigue ---
dat %>% filter(K == "1", fatigue_alpha == 0) %>%
  group_by(variant, theta_f, strategy) %>%
  summarise(ross = mean(ross), .groups = "drop") %>%
  pivot_wider(names_from = strategy, values_from = ross) %>%
  print(n = Inf)

