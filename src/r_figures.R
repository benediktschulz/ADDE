## Script for evaluation of aggregation methods of NN methods

#### Housekeeping ####
rm(list=ls())
gc()

#### Settings ####
# Load package
library(scoringRules)
library(dplyr)
library(RColorBrewer)
library(ggplot2)
library(gridExtra)

# Path for repository (Needs to be adapted)
git_path <- "path_to_git_repo"

# Path for r files
r_path <- paste0(git_path, "r_scripts/")

# Path for figures
pdf_path <- paste0(git_path, "plots/")

# Load functions
source(file = paste0(r_path, "/fn_basic.R"))
source(file = paste0(r_path, "/fn_eval.R"))

#### Initialize ####
# Vector for plotting on [0,1]
x_plot <- seq(0, 1, 0.001)

# Vector for plotting on [0,50]
x_plot50 <- seq(0, 50, 0.01)

# Network types
nn_vec <- c("drn", "bqn", "hen")

# Names of aggregation methods
agg_names <- c(
  "lp" = "Linear Pool",
  "vi" = "Vincentization",
  "vi-a" = "Vincentization (a)",
  "vi-w" = "Vincentization (w)",
  "vi-aw" = "Vincentization (a, w)"
)

# Names of aggregation methods
agg_abr <- c(
  "lp" = expression("LP"),
  "vi" = expression("V[0]^\"=\""),
  "vi-a" = expression("V[a]^\"=\""),
  "vi-w" = expression("V[0]^w"),
  "vi-aw" = expression("V[a]^w")
)
agg_abr <- c(
  "lp" = "LP",
  "vi" = "V[0]^\"=\"",
  "vi-a" = "V[a]^\"=\"",
  "vi-w" = "V[0]^w",
  "vi-aw" = "V[a]^w"
)

# Aggregation methods
agg_meths <- names(agg_names)

# Get colors
cols <- brewer.pal(n = 8,
                   name = "Dark2")

# Colors of aggregation methods
agg_col <- c(
  "lp" = cols[6],
  "vi" = cols[5],
  "vi-a" = cols[1],
  "vi-w" = cols[3],
  "vi-aw" = cols[4],
  "ens" = cols[8],
  "opt" = cols[7]
)

# Line types of aggregation methods
agg_lty <- c(
  "lp" = 2,
  "vi" = 1,
  "vi-a" = 1,
  "vi-w" = 1,
  "vi-aw" = 1,
  "ens" = 4,
  "opt" = 4
)

#### Section 2: Example aggregation ####
# Line width for plotting
lwd_plot <- 3

# Sizes
cex_main <- 1.3
cex_lab <- 1.2
cex_axis <- 1.2

# Evaluations for plotting
n_plot <- 1e+3

# Aggregation methods to plot
agg_meths_plot <- c("lp", "vi", "vi-a", "vi-w", "vi-aw")

# Number of forecasts to aggregate
n <- 2

# Lower and upper boundaries
l <- 0.5
u <- 13.5

# Parameters of individual distributions
mu_1 <- 7
mu_2 <- 10
sd_1 <- 1
sd_2 <- 1

# Data frame of distributions
df_plot <- data.frame(y = numeric(length = n_plot))

# Add probabilities and densities
for(temp in c("1", "2", agg_meths_plot)){
  df_plot[[paste0("p_", temp)]] <- df_plot[[paste0("d_", temp)]] <- 
    numeric(length = n_plot) }

# Vector to plot on
df_plot[["y"]] <- seq(from = l,
                      to = u,
                      length.out = n_plot)

# Calculate probabilities and densities of individual forecasts
for(temp in 1:2){
  df_plot[[paste0("p_", temp)]] <- pnorm(q = df_plot[["y"]],
                                         mean = get(paste0("mu_", temp)),
                                         sd = get(paste0("sd_", temp)))
  df_plot[[paste0("d_", temp)]] <- dnorm(x = df_plot[["y"]],
                                         mean = get(paste0("mu_", temp)),
                                         sd = get(paste0("sd_", temp)))
}

# LP
df_plot[["p_lp"]] <- rowMeans(df_plot[,paste0("p_", 1:2)])
df_plot[["d_lp"]] <- rowMeans(df_plot[,paste0("d_", 1:2)])

# For-Loop over Vincentization approaches
for(temp in agg_meths_plot[grepl("vi", agg_meths_plot, fixed = TRUE)]){
  # Intercept
  if(is.element(temp, c("vi", "vi-w"))){ a <- 0 }
  else if(is.element(temp, c("vi-a", "vi-aw"))){ a <- -6 }
  
  # Weights
  if(is.element(temp, c("vi", "vi-a"))){ w <- 1/n }
  else if(is.element(temp, c("vi-w", "vi-aw"))){ w <- 1/n + 0.15 }
  
  # Calculate mean and standard deviation
  mu_vi <- a + w*sum(c(mu_1, mu_2))
  sd_vi <- w*sum(c(sd_1, sd_2))
  
  # Calculate probabilities and densities
  df_plot[[paste0("p_", temp)]] <- pnorm(q = df_plot[["y"]],
                                         mean = mu_vi,
                                         sd = sd_vi)
  df_plot[[paste0("d_", temp)]] <- dnorm(x = df_plot[["y"]],
                                         mean = mu_vi,
                                         sd = sd_vi)
}

# Name of PDF
file_pdf <- paste0(pdf_path, "/aggregation_methods.pdf")

# Start PDF
pdf(file = file_pdf,
    height = 8,
    width = 25,
    pointsize = 28)

# Set margins
par(mfrow = c(1, 3),
    oma = c(4, 1, 0, 1),
    mar = c(2, 2, 3, 1))

## PDF
# Empty plot
plot(x = 0,
     y = 0,
     type = "n",
     xlab = "y",
     ylab = "f(y)",
     main = "Probability density function (PDF)",
     # main = "PDF",
     cex.axis = cex_axis,
     cex.lab = cex_lab,
     cex.main = cex_main,
     ylim = c(0, max(df_plot[,grepl("d_", colnames(df_plot), fixed = TRUE)])),
     xlim = c(l, u))

# Draw individual PDFs
for(i in 1:n){
  lines(x = df_plot[["y"]],
        y = df_plot[[paste0("d_", i)]],
        col = agg_col["ens"],
        lty = agg_lty["ens"],
        cex = lwd_plot,
        pch = lwd_plot,
        lwd = lwd_plot) 
}

# For-Loop over aggregation methods
for(temp in agg_meths_plot){
  lines(x = df_plot[["y"]],
        y = df_plot[[paste0("d_", temp)]],
        col = agg_col[temp],
        lty = agg_lty[temp],
        lwd = lwd_plot)
}

## CDF
# Empty plot
plot(x = 0,
     y = 0,
     type = "n",
     xlab = "y",
     ylab = "F(y)",
     main = "Cumulative distribution function (CDF)",
     # main = "CDF",
     cex.axis = cex_axis,
     cex.lab = cex_lab,
     cex.main = cex_main,
     ylim = c(0, 1),
     xlim = c(l, u))

# Draw individual CDFs
for(i in 1:n){
  lines(x = df_plot[["y"]],
        y = df_plot[[paste0("p_", i)]],
        col = agg_col["ens"],
        lty = agg_lty["ens"],
        lwd = lwd_plot) 
}

# For-Loop over aggregation methods
for(temp in agg_meths_plot){
  lines(x = df_plot[["y"]],
        y = df_plot[[paste0("p_", temp)]],
        col = agg_col[temp],
        lty = agg_lty[temp],
        lwd = lwd_plot)
}

## Quantile functions
# Empty plot
plot(x = 0,
     y = 0,
     type = "n",
     xlab = "p",
     ylab = "Q(p)",
     main = "Quantile function",
     cex.axis = cex_axis,
     cex.lab = cex_lab,
     cex.main = cex_main,
     xlim = c(0, 1),
     ylim = c(l, u))

# Draw individual quantile functions
for(i in 1:n){
  lines(y = df_plot[["y"]],
        x = df_plot[[paste0("p_", i)]],
        col = agg_col["ens"],
        lty = agg_lty["ens"],
        lwd = lwd_plot) 
}

# For-Loop over aggregation methods
for(temp in agg_meths_plot){
  lines(y = df_plot[["y"]],
        x = df_plot[[paste0("p_", temp)]],
        col = agg_col[temp],
        lty = agg_lty[temp],
        lwd = lwd_plot)
}

# Set margins
par(fig = c(0, 1, 0, 1),
    oma = c(0, 0, 0, 0),
    mar = c(0, 0, 0, 0),
    new = TRUE)

# Empty Plot
plot(x = 0,
     y = 0,
     type = 'l',
     bty = 'n',
     axes = FALSE)

# Add Legend
legend(x = "bottom",
       bty = "n",
       horiz = TRUE,
       inset = 0.01,
       xpd = TRUE,
       cex = 1.5,
       legend = as.expression(parse(text = c("F[1]", "F[2]", agg_abr))),
       col = c(rep(agg_col["ens"], 2), agg_col[agg_meths]),
       lty = c(rep(agg_lty["ens"], 2), agg_lty[agg_meths]),
       lwd = lwd_plot + 2)

# End PDF
dev.off()

#### Section 2: Aggregation of HEN ####
# Line width for plotting
lwd_plot <- 3

# Sizes
cex_main <- 1.7
cex_lab <- 1.7
cex_axis <- 1.7

# Number of forecasts to aggregate
n <- 2

# Number of bins
n_bins <- 8

# Parameters of underlying normal distribution
mu_1 <- n_bins/2 - 0.75
mu_2 <- n_bins/2 + 0.75
sd_1 <- 1
sd_2 <- 1

# Define binning
bin_edges <- seq(from = 0,
                 to = n_bins,
                 by = 1)

# Via normal distribution
p_bins <- rbind(dnorm(x = bin_edges[-1],
                      mean = mu_1,
                      sd = sd_1),
                dnorm(x = bin_edges[-1],
                      mean = mu_2,
                      sd = sd_2))

# Standardize probabilities
p_bins <- p_bins/rowSums(p_bins)

# Get accumulated sums
p_cum <- t(apply(p_bins, 1, cumsum))

# Get accumulated sums of aggregated forecast    
p_cum_sort <- unique(c(0, round(sort(as.vector(p_cum)), 4)))

# Generate corresponding bin probabilities
p_bins_f <- diff(p_cum_sort)

# Generate bin edges for each forecast
bin_edges_f <- rowMeans(sapply(1:n, function(i_sim){
  quant_hd(tau = p_cum_sort,
           probs = p_bins[i_sim,],
           bin_edges = bin_edges) }))

# Number of bins for aggregated forecast
n_bins_f <- length(bin_edges_f) - 1

# Name of PDF
file_pdf <- paste0(pdf_path, "/aggregation_hen.pdf")

# Start PDF
pdf(file = file_pdf,
    height = 9,
    width = 28,
    pointsize = 25)

# Plot together in two plots
par(mfrow = c(1, 3))

### PDF
# Empty plot
plot(x = 0,
     y = 0,
     xlim = range(bin_edges),
     ylim = c(0, max(c(p_bins, p_bins_f))),
     type = "n",
     main = "Probability density function (PDF)",
     # main = "PDF",
     cex.axis = cex_axis,
     cex.lab = cex_lab,
     cex.main = cex_main,
     xlab = "y",
     ylab = "f(y)")

# Axis

# For-Loop over individual forecasts
for(j in 1:n){ 
  # Standardize bin probabilities with length
  p_tilde <- p_bins[j,]/diff(bin_edges)
  
  # Color
  col_alpha <- col2rgb(agg_col["ens"])
  
  # For-Loop over bins
  for(i in 2:length(bin_edges)){
    # Rectangle for each bin
    rect(xleft = bin_edges[i-1],
         xright = bin_edges[i],
         ybottom = 0,
         ytop = p_tilde[i-1],
         col = rgb(col_alpha[1], col_alpha[2], col_alpha[3], 
                   max = 255, alpha = 50 + 50*(j-1)),
         lwd = 0,
         border = NA)
    
    # Draw borders
    lines(x = bin_edges[c(i-1, i-1, i)],
          y = c(0, p_tilde)[c(i-1, i, i)],
          lwd = lwd_plot,
          col = agg_col["ens"])
  }
  
  # close last box
  lines(x = bin_edges[rep((n_bins + 1), 2)],
        y = c(p_tilde[n_bins], 0),
        lwd = lwd_plot,
        col = agg_col["ens"])
}

# V_0^=
# Standardize bin probabilities with length
p_tilde <- p_bins_f/diff(bin_edges_f)

# Color
col_alpha <- col2rgb(agg_col["vi"])

# For-Loop over bins
for(i in 2:length(bin_edges_f)){
  # Rectangle for each bin
  rect(xleft = bin_edges_f[i-1],
       xright = bin_edges_f[i],
       ybottom = 0,
       ytop = p_tilde[i-1],
       col = rgb(col_alpha[1], col_alpha[2], col_alpha[3], 
                 max = 255, alpha = 100),
       lwd = 0,
       border = NA)
  
  # Draw borders
  lines(x = bin_edges_f[c(i-1, i-1, i)],
        y = c(0, p_tilde)[c(i-1, i, i)],
        lwd = lwd_plot,
        # lty = agg_lty["vi"],
        col = agg_col["vi"])
}

# close last box
lines(x = bin_edges_f[rep((n_bins_f + 1), 2)],
      y = c(p_tilde[n_bins_f], 0),
      lwd = lwd_plot,
      col = agg_col["vi"])

# LP
# Standardize bin probabilities with length
p_tilde <- colMeans(p_bins)/diff(bin_edges)

# Color
col_alpha <- col2rgb(agg_col["lp"])

# For-Loop over bins
for(i in 2:length(bin_edges)){
  # Rectangle for each bin
  rect(xleft = bin_edges[i-1],
       xright = bin_edges[i],
       ybottom = 0,
       ytop = p_tilde[i-1],
       col = rgb(col_alpha[1], col_alpha[2], col_alpha[3], 
                 max = 255, alpha = 100),
       lwd = 0,
       border = NA)
  
  # Draw borders
  lines(x = bin_edges[c(i-1, i-1, i)],
        y = c(0, p_tilde)[c(i-1, i, i)],
        lwd = lwd_plot,
        # lty = agg_lty["lp"],
        col = agg_col["lp"])
}

# close last box
lines(x = bin_edges[rep((n_bins + 1), 2)],
      y = c(p_tilde[n_bins], 0),
      lwd = lwd_plot,
      col = agg_col["lp"])

### CDF
# Empty plot
plot(x = 0,
     y = 0,
     type = "n",
     xlab = "y",
     ylab = "F(y)",
     main = "Cumulative distribution function (CDF)",
     # main = "CDF",
     cex.axis = cex_axis,
     cex.lab = cex_lab,
     cex.main = cex_main,
     ylim = c(0, 1),
     xlim = range(bin_edges))

# Draw vertical lines at bin edges
abline(v = bin_edges,
       lty = 2,
       col = "lightgrey")

# Draw individual CDFs
for(i in 1:n){
  lines(x = bin_edges,
        y = c(0, p_cum[i,]),
        col = agg_col["ens"],
        # lty = agg_lty["ens"],
        lwd = lwd_plot)
  points(x = bin_edges,
         y = c(0, p_cum[i,]),
         col = agg_col["ens"],
         pch = 4,
         lwd = lwd_plot)
}

# Draw vertical lines at bin edges for LP
for(i in 2:(n_bins + 1)){
  lines(x = rep(bin_edges[i], 2),
        y = p_cum[,(i-1)],
        lwd = lwd_plot - 1,
        lty = 2,
        col = agg_col["lp"]) }

# V_0^=
lines(x = bin_edges_f,
      y = p_cum_sort,
      col = agg_col["vi"],
      # lty = agg_lty["vi"],
      lwd = lwd_plot)
points(x = bin_edges_f,
       y = p_cum_sort,
       col = agg_col["vi"],
       pch = 4,
       lwd = lwd_plot)

# LP
lines(x = bin_edges,
      y = c(0, colMeans(p_cum)),
      col = agg_col["lp"],
      # lty = agg_lty["lp"],
      lwd = lwd_plot)
points(x = bin_edges,
       y = c(0, colMeans(p_cum)),
       col = agg_col["lp"],
       pch = 4,
       lwd = lwd_plot)

# Legend
legend(x = "bottomright",
       cex = 1.8,
       legend = as.expression(parse(text = c("F[1]", "F[2]", agg_abr[c("lp", "vi")]))),
       col = c(rep(agg_col["ens"], 2), agg_col[c("lp", "vi")]),
       lwd = lwd_plot)

### Quantile functions
# Empty plot
plot(x = 0,
     y = 0,
     type = "n",
     xlab = "p",
     ylab = "Q(p)",
     main = "Quantile function",
     xlim = c(0, 1),
     cex.axis = cex_axis,
     cex.lab = cex_lab,
     cex.main = cex_main,
     ylim = range(bin_edges))

# Draw vertical lines at bin edges
abline(v = p_cum_sort,
       lty = 2,
       col = "lightgrey")

# Draw individual CDFs
for(i in 1:n){
  lines(y = bin_edges,
        x = c(0, p_cum[i,]),
        col = agg_col["ens"],
        lwd = lwd_plot)
  points(y = bin_edges,
         x = c(0, p_cum[i,]),
         col = agg_col["ens"],
         pch = 4,
         lwd = lwd_plot)
}

# Draw vertical lines at bin edges for V_0^=
for(i in 2:length(p_cum_sort)){
  lines(x = rep(p_cum_sort[i], 2),
        y = sapply(1:n, function(l){ 
          quant_hd(tau = p_cum_sort[i],
                   probs = p_bins[l,],
                   bin_edges = bin_edges) }),
        lty = 2,
        lwd = lwd_plot - 1,
        col = agg_col["vi"]) }

# V_0^=
lines(y = bin_edges_f,
      x = p_cum_sort,
      col = agg_col["vi"],
      lwd = lwd_plot)
points(y = bin_edges_f,
       x = p_cum_sort,
       col = agg_col["vi"],
       pch = 4,
       lwd = lwd_plot)

# LP
lines(y = bin_edges,
      x = c(0, colMeans(p_cum)),
      col = agg_col["lp"],
      lwd = lwd_plot)
points(y = bin_edges,
       x = c(0, colMeans(p_cum)),
       col = agg_col["lp"],
       pch = 4,
       lwd = lwd_plot)

# End PDF
dev.off()

