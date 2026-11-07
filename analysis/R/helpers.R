# Shared setup for the analysis scripts.
#
# Every script in analysis/ assumes that the working directory is the
# analysis/ folder: open analysis/analysis.Rproj in RStudio, or run
#   cd analysis && Rscript 01_descriptives.R

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(sandwich)
  library(lmtest)
})

if (!dir.exists(file.path("..", "data", "results"))) {
  stop("Run the analysis scripts with the working directory set to the 'analysis' ",
       "folder (open analysis/analysis.Rproj).", call. = FALSE)
}

RESULTS_DIR <- file.path("..", "data", "results")
OUTPUT_DIR  <- "output"
FIG_DIR     <- file.path("..", "figures", "paper")
dir.create(OUTPUT_DIR, showWarnings = FALSE)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

CASCADE_CUTOFF <- 0.10          # global cascade = at least 10% of accounts adopt
STRATEGIES     <- c("Mega", "Micro", "Niche", "Random")
options(width = 120, dplyr.summarise.inform = FALSE)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Reads one results file (stage 2 or later) and adds the analysis variables.
#   theta_u : mean threshold in steps of 0.05, centred at 0.15
#   alpha_u : fatigue rate in steps of 0.05, centred at 0.05
#   cond    : one experimental condition (campaign size x threshold x fatigue)
read_results <- function(file) {
  read.csv(file.path(RESULTS_DIR, file)) %>%
    mutate(
      strategy = factor(strategy, levels = intersect(STRATEGIES, unique(strategy))),
      cascade  = market_saturation >= CASCADE_CUTOFF,
      K        = factor(n_megas, levels = sort(unique(n_megas))),
      theta_f  = factor(mean_threshold_nominal),
      alpha_f  = factor(fatigue_alpha),
      theta_u  = (mean_threshold_nominal - 0.15) / 0.05,
      alpha_u  = (fatigue_alpha - 0.05) / 0.05,
      cond     = interaction(K, theta_f, alpha_f, drop = TRUE)
    )
}

# Main experiment (stage 3): 9,600 runs, 50 replications.
read_main <- function() read_results("03_main_experiment.csv")

# Rewiring series without fatigue and with 30 replications:
# 0% (stage 3), 10/25/50% (stage 5) and 100% (stage 4).
read_rewiring_series <- function() {
  base <- read.csv(file.path(RESULTS_DIR, "03_main_experiment.csv")) %>%
    filter(replication <= 30, fatigue_alpha == 0) %>% mutate(variant = "rewire0")
  part <- read.csv(file.path(RESULTS_DIR, "05_partial_rewiring.csv"))
  full <- read.csv(file.path(RESULTS_DIR, "04_robustness.csv")) %>%
    filter(variant == "rewired", fatigue_alpha == 0) %>% mutate(variant = "rewire100")
  bind_rows(base, part, full) %>%
    mutate(
      pct      = as.numeric(sub("rewire", "", variant)),
      pct_f    = factor(pct, levels = c(0, 10, 25, 50, 100)),
      strategy = factor(strategy, levels = STRATEGIES),
      cascade  = market_saturation >= CASCADE_CUTOFF,
      K        = factor(n_megas),
      theta_f  = factor(mean_threshold_nominal),
      theta_u  = (mean_threshold_nominal - 0.15) / 0.05
    )
}


# ---------------------------------------------------------------------------
# Proportions with Wilson 95% intervals (they stay inside [0, 1] and do not
# collapse to zero width when no run, or every run, produced a cascade)
# ---------------------------------------------------------------------------
wilson_lo <- function(x, n, z = 1.96) {
  p <- x / n
  (p + z^2 / (2 * n) - z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2))) / (1 + z^2 / n)
}
wilson_hi <- function(x, n, z = 1.96) {
  p <- x / n
  (p + z^2 / (2 * n) + z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2))) / (1 + z^2 / n)
}

# Cascade counts and rates with Wilson intervals, for any grouping.
cascade_rates <- function(d, ...) {
  d %>%
    group_by(...) %>%
    summarise(cascades = sum(cascade), runs = n(), .groups = "drop") %>%
    mutate(rate = cascades / runs,
           lo   = wilson_lo(cascades, runs),
           hi   = wilson_hi(cascades, runs))
}


# ---------------------------------------------------------------------------
# Strategy models
#
# Logistic regression of the cascade indicator on strategy with a fixed effect
# for every condition (`fe`); standard errors clustered by replication
# (sandwich::vcovCL, type HC0 with the G/(G-1) small-sample adjustment).
# Conditions in which no run or every run cascaded carry no information on
# strategy differences and are dropped first (`informative()`).
# ---------------------------------------------------------------------------
informative <- function(d, fe = "cond") {
  keep <- d %>%
    group_by(.data[[fe]]) %>%
    summarise(ok = n_distinct(cascade) == 2, .groups = "drop") %>%
    filter(ok) %>%
    pull(1)
  d %>% filter(.data[[fe]] %in% keep) %>% droplevels()
}

# moderators: NULL, or a character vector such as c("theta_u") or
# c("theta_u", "alpha_u"); each enters as an interaction with strategy.
fit_strategy_model <- function(d, ref = "Micro", moderators = NULL, fe = "cond") {
  d$s <- relevel(droplevels(factor(d$strategy)), ref = ref)
  rhs <- "s"
  for (m in moderators) rhs <- paste0(rhs, " + s * ", m)
  if (!is.null(fe) && nlevels(factor(d[[fe]])) > 1) rhs <- paste(rhs, "+", fe)
  model <- suppressWarnings(glm(as.formula(paste("cascade ~", rhs)),
                                family = binomial, data = d))
  ct <- coeftest(model, vcov = vcovCL(model, cluster = d$replication))
  keep <- grepl("^s[A-Z]", rownames(ct))
  out <- data.frame(
    term = rownames(ct)[keep],
    OR   = exp(ct[keep, 1]),
    lo   = exp(ct[keep, 1] - 1.96 * ct[keep, 2]),
    hi   = exp(ct[keep, 1] + 1.96 * ct[keep, 2]),
    p    = ct[keep, 4],
    row.names = NULL
  )
  out$term <- label_terms(out$term, ref)
  attr(out, "info") <- c(runs = nrow(d),
                         conditions = if (is.null(fe)) 1 else nlevels(factor(d[[fe]])),
                         clusters = length(unique(d$replication)))
  out
}

label_terms <- function(term, ref) {
  term <- sub("^s([A-Za-z]+):theta_u$", "\\1 x threshold (per +0.05)", term)
  term <- sub("^s([A-Za-z]+):alpha_u$", "\\1 x fatigue (per +0.05)", term)
  term <- sub("^s([A-Za-z]+)$", paste0("\\1 vs ", ref), term)
  term
}

# Prints a model table with its sample information.
show_model <- function(tab, title) {
  info <- attr(tab, "info")
  cat("\n", title, "\n", sep = "")
  cat(sprintf("  runs = %d, informative conditions = %d, replication clusters = %d\n",
              info[["runs"]], info[["conditions"]], info[["clusters"]]))
  print(format_or(tab), row.names = FALSE)
  invisible(tab)
}

format_or <- function(tab) {
  tab %>% mutate(across(c(OR, lo, hi), ~ round(.x, 2)), p = signif(p, 2))
}

write_table <- function(tab, file) {
  write.csv(tab, file.path(OUTPUT_DIR, file), row.names = FALSE)
  invisible(tab)
}
