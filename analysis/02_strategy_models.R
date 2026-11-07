# 02_strategy_models.R - strategy comparisons in the main experiment
# (Sections 3.2-3.4 of the paper).
#
# Model (see R/helpers.R):
#   logit Pr(cascade) = delta_c + beta_s [ + gamma_s * moderator ]
#   delta_c : fixed effect for each condition c (campaign size x threshold x fatigue)
#   beta_s  : effect of strategy s relative to the reference strategy
#   moderators: threshold in steps of 0.05 centred at 0.15 (theta_u),
#               fatigue in steps of 0.05 centred at 0.05 (alpha_u),
#               each in a separate model.
# Standard errors are clustered by replication (50 clusters): within a
# replication all strategies and conditions share seeds and thresholds.
# Conditions in which no run or every run cascaded are dropped (32 of 48 remain).
#
# Intervals describe Monte Carlo variation across replications, conditional on
# the model, its parameters and the single network used.
#
# Input : data/results/03_main_experiment.csv
# Output: analysis/output/strategy_models.csv

source(file.path("R", "helpers.R"))
d <- informative(read_main())

models <- list(
  "Pooled contrasts, reference Mega"   = fit_strategy_model(d, ref = "Mega"),
  "Pooled contrasts, reference Micro"  = fit_strategy_model(d, ref = "Micro"),
  "Pooled contrasts, reference Random" = fit_strategy_model(d, ref = "Random"),
  "Strategy x threshold, reference Micro" =
    fit_strategy_model(d, ref = "Micro", moderators = "theta_u"),
  "Strategy x fatigue, reference Micro" =
    fit_strategy_model(d, ref = "Micro", moderators = "alpha_u")
)
for (nm in names(models)) show_model(models[[nm]], nm)

# Model-implied Niche-vs-Micro and Mega-vs-Micro odds ratios at each threshold
# (point estimates from the strategy x threshold model; no intervals).
m <- models[["Strategy x threshold, reference Micro"]]
implied <- expand.grid(strategy = c("Mega", "Niche", "Random"),
                       threshold = c(0.10, 0.15, 0.20, 0.25)) %>%
  rowwise() %>%
  mutate(OR = m$OR[m$term == paste(strategy, "vs Micro")] *
           m$OR[m$term == paste(strategy, "x threshold (per +0.05)")]^((threshold - 0.15) / 0.05)) %>%
  ungroup() %>%
  mutate(OR = round(OR, 2))
cat("\nModel-implied odds ratios vs Micro by threshold (point estimates):\n")
print(implied %>% pivot_wider(names_from = threshold, values_from = OR))

out <- bind_rows(lapply(names(models), function(nm) {
  info <- attr(models[[nm]], "info")
  cbind(model = nm, models[[nm]], runs = info[["runs"]],
        conditions = info[["conditions"]], clusters = info[["clusters"]])
}))
write_table(out, "strategy_models.csv")
cat("\nWritten to output/strategy_models.csv\n")
