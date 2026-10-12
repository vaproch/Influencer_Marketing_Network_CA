rm(list=ls())
graphics.off()
cat("\014")

library(dplyr)
library(tidyr)
library(sandwich)
library(lmtest)
library(ggplot2)

v3 <- read.csv("simulation_results_v3.csv")
v4 <- read.csv("simulation_results_v4_robustness.csv")
v5 <- read.csv("simulation_results_v5_rewiring.csv")

# --- 1. Pairing check: Mega/Micro/Random seeds at 10% rewiring must equal v3 ---
key <- c("replication", "strategy", "n_megas", "fatigue_alpha", "mean_threshold_nominal")
a <- v5 %>% filter(variant == "rewire10", strategy != "Niche") %>%
  select(all_of(key), seed_count, achieved_budget)
b <- v3 %>% filter(strategy != "Niche") %>%
  select(all_of(key), seed_count, achieved_budget)
chk <- inner_join(a, b, by = key, suffix = c("_r10", "_v3"))
nrow(chk)                                                # expect 1080
mean(chk$seed_count_r10 == chk$seed_count_v3)            # expect 1
mean(chk$achieved_budget_r10 == chk$achieved_budget_v3)  # expect 1

# --- 2. Combine 0% (v3), 10/25/50% (v5), 100% (v4); no fatigue, 30 replications ---
base <- v3 %>% filter(replication <= 30, fatigue_alpha == 0) %>% mutate(variant = "rewire0")
full <- v4 %>% filter(variant == "rewired", fatigue_alpha == 0) %>% mutate(variant = "rewire100")

dat <- bind_rows(base, v5, full) %>% mutate(
  pct      = factor(as.numeric(sub("rewire", "", variant)), levels = c(0, 10, 25, 50, 100)),
  strategy = factor(strategy, levels = c("Mega", "Micro", "Niche", "Random")),
  K        = factor(n_megas),
  theta_f  = factor(mean_threshold_nominal),
  cascade  = market_saturation >= 0.10
)
nrow(dat)                                                         # expect 7200
all(table(dat$pct, dat$strategy, dat$theta_f, dat$K) == 30)       # expect TRUE

# --- 3. Do Niche seeds keep their overlap as the network is randomised? (K = 3) ---
dat %>% filter(K == "3") %>%
  group_by(pct, strategy) %>%
  summarise(reach = mean(audience_reach), overlap = mean(audience_overlap), .groups = "drop") %>%
  print(n = Inf)

# --- 4. Cascade rate by share of randomised edges (pooled over K) ---
dat %>% group_by(pct, theta_f, strategy) %>%
  summarise(rate = mean(cascade), .groups = "drop") %>%
  pivot_wider(names_from = strategy, values_from = rate) %>%
  print(n = Inf)

# --- 5. Odds of a cascade relative to Micro, by share of randomised edges ---
res <- list()
for (p in levels(dat$pct)) {
  s <- dat %>% filter(pct == p) %>% mutate(cond = interaction(K, theta_f, drop = TRUE))
  info <- s %>% group_by(cond) %>%
    summarise(ok = n_distinct(cascade) == 2, .groups = "drop") %>%
    filter(ok) %>% pull(cond)
  d <- s %>% filter(cond %in% info) %>% droplevels()
  if (nrow(d) == 0) next
  d$s <- relevel(d$strategy, ref = "Micro")
  f <- if (nlevels(d$cond) > 1) cascade ~ s + cond else cascade ~ s
  m <- tryCatch(suppressWarnings(glm(f, family = binomial, data = d)), error = function(e) NULL)
  if (is.null(m)) next
  ct <- tryCatch(coeftest(m, vcov = vcovCL(m, cluster = d$replication)), error = function(e) NULL)
  if (is.null(ct)) next
  o <- data.frame(pct_randomised = p, term = rownames(ct), OR = exp(ct[, 1]),
                  lo = exp(ct[, 1] - 1.96 * ct[, 2]), hi = exp(ct[, 1] + 1.96 * ct[, 2]),
                  p = ct[, 4], n_conditions = nlevels(d$cond))
  res[[p]] <- o[grepl("^s", o$term), ]
}
tab <- bind_rows(res)
tab[, c("OR", "lo", "hi")] <- round(tab[, c("OR", "lo", "hi")], 2)
tab$p <- round(tab$p, 3)
rownames(tab) <- NULL
print(tab, row.names = FALSE)

# --- 6. Does Niche's advantage grow with threshold, at each clustering level? ---
for (p in levels(dat$pct)) {
  s <- dat %>% filter(pct == p) %>%
    mutate(cond = interaction(K, theta_f, drop = TRUE),
           theta_u = (mean_threshold_nominal - 0.15) / 0.05)
  info <- s %>% group_by(cond) %>%
    summarise(ok = n_distinct(cascade) == 2, .groups = "drop") %>%
    filter(ok) %>% pull(cond)
  d <- s %>% filter(cond %in% info) %>% droplevels()
  d$s <- relevel(d$strategy, ref = "Micro")
  cat("\n=== Share randomised:", p, "% | informative conditions:", nlevels(d$cond), "===\n")
  out <- tryCatch({
    m  <- suppressWarnings(glm(cascade ~ s * theta_u + cond, family = binomial, data = d))
    ct <- coeftest(m, vcov = vcovCL(m, cluster = d$replication))
    o  <- data.frame(term = rownames(ct), OR = exp(ct[, 1]),
                     lo = exp(ct[, 1] - 1.96 * ct[, 2]),
                     hi = exp(ct[, 1] + 1.96 * ct[, 2]), p = ct[, 4])
    o <- o[grepl("theta_u", o$term) & grepl("^s", o$term), ]
    o[, -1] <- round(o[, -1], 3)
    rownames(o) <- NULL
    o
  }, error = function(e) paste("model failed:", conditionMessage(e)))
  print(out)
}

# --- 7. Figure: cascade probability against share of randomised edges ---
f4 <- dat %>%
  mutate(K_lab  = factor(paste0("K = ", K), levels = paste0("K = ", c(1, 3, 10))),
         th_lab = factor(paste0("theta = ", mean_threshold_nominal))) %>%
  group_by(K_lab, th_lab, x = as.numeric(as.character(pct)), strategy) %>%
  summarise(p = mean(cascade), n = n(), .groups = "drop") %>%
  mutate(se = sqrt(p * (1 - p) / n), lo = pmax(0, p - 1.96 * se), hi = pmin(1, p + 1.96 * se))

fig4 <- ggplot(f4, aes(x, p, colour = strategy, shape = strategy)) +
  geom_ribbon(aes(ymin = lo, ymax = hi, fill = strategy), alpha = 0.10, colour = NA) +
  geom_line() + geom_point(size = 1.8) +
  facet_grid(th_lab ~ K_lab) +
  scale_x_continuous(breaks = c(0, 10, 25, 50, 100)) +
  labs(x = "Share of edges randomised (%)", y = "Probability of a global cascade",
       colour = "Strategy", shape = "Strategy", fill = "Strategy") +
  theme_bw()
ggsave("fig4_rewiring.png", fig4, width = 10, height = 8, dpi = 300)