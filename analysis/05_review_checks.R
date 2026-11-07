# 05_review_checks.R - additional checks requested during peer review
#
#   1. Budget matching: distribution of actual spend and sensitivity of the
#      strategy comparisons to tighter matching
#   2. Alternative model specifications (replication fixed effects,
#      conditional logit)
#   3. Counts and denominators behind every cell-level number quoted in the
#      paper, with the aggregation stated explicitly
#
# Input : data/results/03_main_experiment.csv, 04_robustness.csv, 05_partial_rewiring.csv
# Output: analysis/output/review_*.csv

source(file.path("R", "helpers.R"))
d <- read_main()

# ===========================================================================
# 1. Budget matching
# ===========================================================================
# Mega spends the budget exactly. The other strategies add accounts until the
# spend is within +-2% of the budget, skipping accounts that would exceed it;
# because accounts are discrete, the last gap cannot always be filled.
seed_sets <- d %>%
  distinct(replication, n_megas, strategy, achieved_budget, target_budget) %>%
  mutate(spend_share = achieved_budget / target_budget)

spend <- seed_sets %>%
  group_by(strategy) %>%
  summarise(seed_sets = n(),
            mean = round(mean(spend_share), 4),
            min = round(min(spend_share), 4),
            max = round(max(spend_share), 4),
            below_98pct = sum(spend_share < 0.98),
            above_102pct = sum(spend_share > 1.02))
spend_by_K <- seed_sets %>%
  group_by(n_megas, strategy) %>%
  summarise(seed_sets = n(),
            mean = round(mean(spend_share), 4),
            min = round(min(spend_share), 4),
            below_98pct = sum(spend_share < 0.98), .groups = "drop")
cat("Actual spend as a share of the budget (one row per strategy, 150 seed sets each):\n")
print(as.data.frame(spend), row.names = FALSE)
cat("\nBy campaign size:\n")
print(as.data.frame(spend_by_K), row.names = FALSE)
write_table(spend, "review_spend_distribution.csv")
write_table(spend_by_K, "review_spend_distribution_by_K.csv")

# Replication x campaign-size combinations in which every strategy spent
# between 98% and 102% of the budget
tight <- seed_sets %>%
  group_by(replication, n_megas) %>%
  summarise(tight = all(spend_share >= 0.98 & spend_share <= 1.02), .groups = "drop")
cat(sprintf("\nTightly matched replication x K combinations: %d of %d\n",
            sum(tight$tight), nrow(tight)))

d_tight <- d %>% left_join(tight, by = c("replication", "n_megas")) %>% filter(tight)

sensitivity <- list(
  "All runs (as in the paper)"             = informative(d),
  "Tightly matched budgets only (98-102%)" = informative(d_tight),
  "Campaign sizes K = 3 and 10 only"       = informative(d %>% filter(n_megas > 1))
)
sens_out <- bind_rows(lapply(names(sensitivity), function(nm) {
  dd <- sensitivity[[nm]]
  tabs <- list(fit_strategy_model(dd, ref = "Mega"),
               fit_strategy_model(dd, ref = "Micro"),
               fit_strategy_model(dd, ref = "Micro", moderators = "theta_u"),
               fit_strategy_model(dd, ref = "Micro", moderators = "alpha_u"))
  tabs[[3]] <- tabs[[3]][grepl("threshold", tabs[[3]]$term), ]
  tabs[[4]] <- tabs[[4]][grepl("fatigue", tabs[[4]]$term), ]
  tab <- bind_rows(tabs)
  attr(tab, "info") <- attr(fit_strategy_model(dd, ref = "Micro"), "info")
  show_model(tab, paste("Sensitivity:", nm))
  cbind(sample = nm, tab)
}))
write_table(sens_out, "review_budget_sensitivity.csv")

# ===========================================================================
# 2. Alternative model specifications
# ===========================================================================
di <- informative(d)
cat("\n--- Alternative specifications (reference Micro) ---\n")
show_model(fit_strategy_model(di, ref = "Micro"),
           "Condition fixed effects, clustered by replication (as in the paper)")

d_rep <- di %>% mutate(rep_f = factor(replication))
m_rep <- suppressWarnings(glm(cascade ~ relevel(strategy, ref = "Micro") + cond + rep_f,
                              family = binomial, data = d_rep))
ct <- coeftest(m_rep, vcov = vcovCL(m_rep, cluster = d_rep$replication))
rows <- grepl("^relevel", rownames(ct))
cat("\nCondition + replication fixed effects, clustered by replication:\n")
print(data.frame(term = sub("^relevel\\(strategy, ref = \"Micro\"\\)", "", rownames(ct)[rows]),
                 OR = round(exp(ct[rows, 1]), 2),
                 lo = round(exp(ct[rows, 1] - 1.96 * ct[rows, 2]), 2),
                 hi = round(exp(ct[rows, 1] + 1.96 * ct[rows, 2]), 2)), row.names = FALSE)

if (requireNamespace("survival", quietly = TRUE)) {
  suppressPackageStartupMessages(library(survival))
  # Conditional logit: each condition x replication is a stratum of four
  # strategies facing identical thresholds; robust SEs clustered by replication.
  cl <- tryCatch(
    clogit(cascade ~ relevel(strategy, ref = "Micro") +
             strata(cond, replication) + cluster(replication),
                     data = di, method = "efron"),
    error = function(e) e)
  if (inherits(cl, "error")) {
    cat("\nConditional logit could not be fitted:", conditionMessage(cl), "\n")
  } else {
    s <- summary(cl)$conf.int
    cat("\nConditional logit (strata = condition x replication, robust SEs by replication):\n")
    print(data.frame(term = sub("^relevel\\(strategy, ref = \"Micro\"\\)", "", rownames(s)),
                     OR = round(s[, 1], 2), lo = round(s[, 3], 2), hi = round(s[, 4], 2)),
          row.names = FALSE)
  }
}

# ===========================================================================
# 3. Counts behind the cell-level numbers quoted in the paper
# ===========================================================================
# Each block states the conditions and the aggregation; `cascades` / `runs`
# are the success counts and denominators.
audit <- function(data, statement, aggregation, ...) {
  data %>%
    group_by(...) %>%
    summarise(cascades = sum(cascade), runs = n(), .groups = "drop") %>%
    mutate(rate = round(cascades / runs, 3),
           statement = statement, aggregation = aggregation, .before = 1)
}

price <- read_results("04_robustness.csv")
rw <- read_rewiring_series()

quoted <- bind_rows(
  audit(d %>% filter(n_megas == 1, mean_threshold_nominal == 0.10, fatigue_alpha == 0),
        "3.2: K = 1, mu = 0.10, no fatigue", "single cell", strategy),
  audit(d %>% filter(n_megas == 1, mean_threshold_nominal == 0.20, fatigue_alpha == 0),
        "3.4: K = 1, mu = 0.20, no fatigue", "single cell", strategy),
  audit(d %>% filter(n_megas == 10, mean_threshold_nominal == 0.25, fatigue_alpha == 0),
        "3.4: K = 10, mu = 0.25, no fatigue", "single cell", strategy),
  audit(d %>% filter(n_megas == 3, mean_threshold_nominal == 0.10),
        "3.4: K = 3, mu = 0.10", "pooled over the 4 fatigue levels", strategy),
  audit(d %>% filter(n_megas == 3, mean_threshold_nominal == 0.10),
        "3.4: K = 3, mu = 0.10, by fatigue", "single cells", fatigue_alpha, strategy),
  audit(d %>% filter(n_megas == 10, fatigue_alpha == 0),
        "3.1: K = 10, no fatigue, by threshold", "single cells", mean_threshold_nominal, strategy),
  audit(d %>% filter(n_megas == 10, mean_threshold_nominal == 0.20, fatigue_alpha == 0.05),
        "3.1: K = 10, mu = 0.20, fatigue 0.05", "single cell", strategy),
  audit(price %>% filter(variant == "gamma0.7", mean_threshold_nominal == 0.20, fatigue_alpha == 0),
        "3.5: price exponent 0.7, mu = 0.20, no fatigue", "pooled over K = 1, 3, 10", strategy),
  audit(price %>% filter(variant == "gamma0.7", mean_threshold_nominal == 0.20, fatigue_alpha == 0),
        "3.5: price exponent 0.7, mu = 0.20, no fatigue, by K", "single cells", n_megas, strategy),
  audit(rw %>% filter(strategy == "Mega", mean_threshold_nominal == 0.15),
        "3.5: Mega, mu = 0.15, no fatigue, by share randomised", "pooled over K = 1, 3, 10", pct),
  audit(rw %>% filter(pct == 100),
        "3.5: fully randomised network", "single cells", n_megas, mean_threshold_nominal, strategy) %>%
    arrange(desc(cascades)) %>% head(4)
)
cat("\nCounts behind quoted cascade rates:\n")
print(as.data.frame(quoted), row.names = FALSE)
write_table(quoted, "review_quoted_counts.csv")

ross_quoted <- bind_rows(
  d %>% filter(n_megas == 1, mean_threshold_nominal == 0.20) %>%
    group_by(fatigue_alpha = as.character(fatigue_alpha), strategy) %>%
    summarise(runs = n(), mean_ross = mean(ross), .groups = "drop") %>%
    mutate(statement = "3.4: ROSS, K = 1, mu = 0.20, by fatigue"),
  d %>% filter(n_megas == 1, mean_threshold_nominal == 0.20) %>%
    group_by(fatigue_alpha = "pooled", strategy) %>%
    summarise(runs = n(), mean_ross = mean(ross), .groups = "drop") %>%
    mutate(statement = "3.4: ROSS, K = 1, mu = 0.20, pooled over fatigue"),
  price %>% filter(variant == "gamma0.7", n_megas == 1, mean_threshold_nominal == 0.20,
                   fatigue_alpha == 0) %>%
    group_by(fatigue_alpha = "0", strategy) %>%
    summarise(runs = n(), mean_ross = mean(ross), .groups = "drop") %>%
    mutate(statement = "3.5: ROSS, price exponent 0.7, K = 1, mu = 0.20, no fatigue")
) %>% mutate(mean_ross = round(mean_ross, 2)) %>% relocate(statement)
cat("\nROSS values quoted in the paper (simulated additional adopters per model cost unit):\n")
print(as.data.frame(ross_quoted), row.names = FALSE)
write_table(ross_quoted, "review_quoted_ross.csv")
