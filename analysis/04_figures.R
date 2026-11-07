# 04_figures.R - Figures 2-4 of the paper
#
#   Figure 2: probability of a global cascade by mean adoption threshold,
#             fatigue rate (columns) and campaign size (rows)
#   Figure 3: odds ratios of a global cascade relative to Micro, estimated
#             separately at each threshold (A) and each fatigue rate (B)
#   Figure 4: probability of a global cascade by share of randomised edges
#
# Probabilities are shown with Wilson 95% intervals, which stay inside [0, 1]
# and keep a positive width when no run (or every run) produced a cascade.
# Figure 1 is drawn by figures/scripts/fig1_seeding_strategies.py.
#
# Output: figures/paper/fig2_cascade_probability.png
#         figures/paper/fig3_odds_ratios.png
#         figures/paper/fig4_rewiring.png
#         analysis/output/fig3_odds_ratios.csv (the numbers behind Figure 3)

source(file.path("R", "helpers.R"))
suppressPackageStartupMessages(library(ggplot2))

PALETTE <- c(Mega = "#D55E00", Micro = "#0072B2", Niche = "#009E73", Random = "#7F7F7F")
SHAPES  <- c(Mega = 16, Micro = 17, Niche = 15, Random = 3)

theme_paper <- function() {
  theme_bw(base_size = 10) +
    theme(legend.position = "bottom",
          legend.title = element_blank(),
          panel.spacing = unit(0.9, "lines"),
          strip.background = element_rect(fill = "grey92"))
}
strategy_scales <- list(
  scale_colour_manual(values = PALETTE),
  scale_fill_manual(values = PALETTE),
  scale_shape_manual(values = SHAPES)
)

d <- read_main()

# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------
f2 <- cascade_rates(d, n_megas, mean_threshold_nominal, fatigue_alpha, strategy) %>%
  mutate(K_lab = factor(paste0("K = ", n_megas), levels = paste0("K = ", c(1, 3, 10))),
         a_lab = factor(paste0("α = ", fatigue_alpha),
                        levels = paste0("α = ", sort(unique(fatigue_alpha)))))

fig2 <- ggplot(f2, aes(mean_threshold_nominal, rate, colour = strategy,
                       fill = strategy, shape = strategy)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.12, colour = NA) +
  geom_line() +
  geom_point(size = 1.6) +
  facet_grid(K_lab ~ a_lab) +
  strategy_scales +
  scale_x_continuous(breaks = c(0.10, 0.15, 0.20, 0.25),
                     labels = c("0.10", "0.15", "0.20", "0.25"),
                     expand = expansion(mult = 0.08)) +
  scale_y_continuous(breaks = c(0, 0.5, 1), expand = expansion(mult = 0.06)) +
  labs(x = "Mean adoption threshold μ (product friction)",
       y = "Probability of a global cascade") +
  theme_paper()
ggsave(file.path(FIG_DIR, "fig2_cascade_probability.png"), fig2,
       width = 7, height = 5.2, dpi = 300)

# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------
# One model per slice: strategy + fixed effects for the remaining conditions,
# standard errors clustered by replication, reference Micro.
slice_or <- function(dd, fe_vars) {
  dd$cond <- do.call(interaction, c(unname(as.list(dd[fe_vars])), drop = TRUE))
  dd <- informative(dd)
  if (nrow(dd) == 0) return(NULL)
  tab <- tryCatch(fit_strategy_model(dd, ref = "Micro"), error = function(e) NULL)
  if (is.null(tab)) return(NULL)
  cbind(tab, conditions = attr(tab, "info")[["conditions"]])
}

by_theta <- bind_rows(lapply(sort(unique(d$mean_threshold_nominal)), function(th) {
  r <- slice_or(d %>% filter(mean_threshold_nominal == th), c("K", "alpha_f"))
  if (is.null(r)) NULL else cbind(panel = "A. By mean adoption threshold",
                                  level = sprintf("%.2f", th), r)
}))
by_alpha <- bind_rows(lapply(sort(unique(d$fatigue_alpha)), function(a) {
  r <- slice_or(d %>% filter(fatigue_alpha == a), c("K", "theta_f"))
  if (is.null(r)) NULL else cbind(panel = "B. By fatigue rate",
                                  level = format(a), r)
}))

f3 <- bind_rows(by_theta, by_alpha) %>%
  mutate(strategy = factor(sub(" vs Micro$", "", term), levels = c("Mega", "Niche", "Random")),
         # estimates from (quasi-)separated slices are not meaningful
         keep = is.finite(lo) & is.finite(hi) & hi / lo < 1e4 & OR > 1e-3 & OR < 1e3) %>%
  filter(keep) %>%
  select(-keep)
write_table(f3 %>% mutate(across(c(OR, lo, hi), ~ round(.x, 3))), "fig3_odds_ratios.csv")

fig3 <- ggplot(f3, aes(level, OR, colour = strategy, shape = strategy)) +
  geom_hline(yintercept = 1, linetype = 2, colour = "grey40") +
  geom_pointrange(aes(ymin = lo, ymax = hi), position = position_dodge(width = 0.6),
                  size = 0.35) +
  facet_wrap(~ panel, scales = "free_x") +
  scale_colour_manual(values = PALETTE) +
  scale_shape_manual(values = SHAPES) +
  scale_y_log10(breaks = c(0.1, 0.25, 0.5, 1, 2, 4, 10, 25, 50),
                labels = c("0.1", "0.25", "0.5", "1", "2", "4", "10", "25", "50")) +
  labs(x = NULL, y = "Odds ratio of a global cascade vs. Micro (log scale)") +
  theme_paper()
ggsave(file.path(FIG_DIR, "fig3_odds_ratios.png"), fig3, width = 7, height = 3.8, dpi = 300)

# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------
f4 <- read_rewiring_series() %>%
  filter(mean_threshold_nominal %in% c(0.15, 0.20)) %>%
  cascade_rates(pct, n_megas, mean_threshold_nominal, strategy) %>%
  mutate(K_lab  = factor(paste0("K = ", n_megas), levels = paste0("K = ", c(1, 3, 10))),
         th_lab = paste0("μ = ", sprintf("%.2f", mean_threshold_nominal)))

fig4 <- ggplot(f4, aes(pct, rate, colour = strategy, fill = strategy, shape = strategy)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.12, colour = NA) +
  geom_line() +
  geom_point(size = 1.6) +
  facet_grid(th_lab ~ K_lab) +
  strategy_scales +
  scale_x_continuous(breaks = c(0, 10, 25, 50, 100), expand = expansion(mult = 0.06)) +
  scale_y_continuous(breaks = c(0, 0.5, 1), expand = expansion(mult = 0.06)) +
  labs(x = "Share of edges randomised (%)", y = "Probability of a global cascade") +
  theme_paper()
ggsave(file.path(FIG_DIR, "fig4_rewiring.png"), fig4, width = 7, height = 4.4, dpi = 300)

cat("Figures written to", normalizePath(FIG_DIR), "\n")
