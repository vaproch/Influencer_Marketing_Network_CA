rm(list=ls())
graphics.off()
cat("\014")

df <- read.csv("simulation_results.csv")
summary(df)

# ==============================================================================
# 1. ENVIRONMENT SETUP & DATA PREPROCESSING
# ==============================================================================
library(tidyverse)
library(broom)
library(ggpubr)

# Ensure categorical factors are correctly typed
df_clean <- df %>%
  mutate(
    strategy = factor(strategy, levels = c("Mega", "Micro", "Random")),
    fatigue_alpha = factor(fatigue_alpha),
    mean_threshold = factor(mean_threshold)
  )

# Preview aggregate metric summary by strategy
df_clean %>%
  group_by(strategy) %>%
  summarize(
    mean_sat = mean(market_saturation),
    sd_sat   = sd(market_saturation),
    mean_ross = mean(ross),
    sd_ross   = sd(ross),
    avg_ticks = mean(convergence_tick),
    .groups = "drop"
  )

# ==============================================================================
# 2. INFERENTIAL STATISTICAL MODELING (ANOVA & POST-HOC)
# ==============================================================================

# Three-Way Factorial ANOVA on Return-On-Seed-Spend (ROSS)
anova_ross <- aov(ross ~ strategy * fatigue_alpha * mean_threshold, data = df_clean)
summary(anova_ross)

# Calculate Effect Sizes (Eta-Squared)
cat("\n--- ANOVA Summary Table with Effect Sizes ---\n")
anova_summary <- tidy(anova_ross)
print(anova_summary)

# Post-Hoc Pairwise Comparison (Tukey HSD for Strategy)
cat("\n--- Tukey HSD Strategy Comparison ---\n")
tukey_strategy <- TukeyHSD(anova_ross, "strategy")
print(tukey_strategy)

# Post-Hoc Strategy Performance per Fatigue Level
cat("\n--- Strategy Comparison across Fatigue Levels ---\n")
tukey_interaction <- TukeyHSD(anova_ross, "strategy:fatigue_alpha")
# Filter interaction results to compare Micro vs Mega at each alpha level
tukey_tidy <- tidy(tukey_interaction) %>%
  filter(str_detect(contrast, "Micro:") & str_detect(contrast, "Mega:"))
print(tukey_tidy)


# ==============================================================================
# 3. FIGURE 1: ROSS PERFORMANCE HEATMAP (RQ2 & RQ3)
# ==============================================================================
# Aggregates ROSS across strategy, fatigue, and adoption threshold

df_heatmap <- df_clean %>%
  group_by(strategy, fatigue_alpha, mean_threshold) %>%
  summarize(mean_ross = mean(ross), .groups = "drop")

fig1 <- ggplot(df_heatmap, aes(x = fatigue_alpha, y = mean_threshold, fill = mean_ross)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = round(mean_ross, 2)), color = "black", size = 3.5, fontface = "bold") +
  facet_wrap(~ strategy) +
  scale_fill_gradient2(
    low = "#D95F02", mid = "#FFD700", high = "#1B9E77",
    midpoint = median(df_heatmap$mean_ross),
    name = "Mean ROSS"
  ) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Return-On-Seed-Spend (ROSS) Across Behavioral Parameters",
    subtitle = "Comparing Mega, Micro, and Random strategies across Ad Fatigue (Alpha) and Threshold (Theta)",
    x = "Ad Fatigue Parameter (\u03b1)",
    y = "Mean Adoption Threshold (\u03b8)"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    strip.text = element_text(face = "bold", size = 12),
    legend.position = "bottom"
  )

print(fig1)
ggsave("figure1_ross_heatmap.png", fig1, width = 11, height = 5, dpi = 300)


# ==============================================================================
# 4. FIGURE 2: TIPPING POINT GRADIENT (MICRO ADVANTAGE OVER MEGA)
# ==============================================================================
# Calculates the net ROSS advantage (Micro ROSS - Mega ROSS) across fatigue alpha

df_advantage <- df %>%
  group_by(replication, fatigue_alpha, mean_threshold, strategy) %>%
  summarize(mean_ross = mean(ross), .groups = "drop") %>%
  pivot_wider(names_from = strategy, values_from = mean_ross) %>%
  mutate(micro_advantage = Micro - Mega)

fig2 <- ggplot(df_advantage, aes(x = factor(fatigue_alpha), y = micro_advantage, fill = factor(mean_threshold))) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 0.8) +
  geom_boxplot(alpha = 0.8, outlier.size = 1) +
  scale_fill_brewer(palette = "Blues", name = "Threshold (\u03b8)") +
  theme_bw(base_size = 12) +
  labs(
    title = "Micro-Influencer ROSS Advantage Over Mega-Influencers",
    subtitle = "Values > 0 indicate Micro Portfolio outperformance; Red dashed line represents Parity",
    x = "Ad Fatigue Parameter (\u03b1)",
    y = "Net ROSS Advantage (\u0394 ROSS = Micro - Mega)"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "right"
  )

print(fig2)
ggsave("figure2_tipping_point.png", fig2, width = 9, height = 5, dpi = 300)