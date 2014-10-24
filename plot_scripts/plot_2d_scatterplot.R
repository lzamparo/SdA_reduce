library(ggplot2)
library(dplyr)
library(gridExtra)

# Load the 2d representation of the data (say, from the 10D versions)
setwd("/data/sda_output_data/distances/2d")

# Load the comparators 50 -> 2d data
isomap_2d <- read.csv("isomap_dim50_2d.csv")
lle_2d <- read.csv("lle_dim50_2d.csv")
kpca_2d <- read.csv("kpca_dim50_2d.csv")
pca_2d <- read.csv("pca_dim50_2d.csv")

# Load the SdA 10 -> 2d data
# one <- read.csv("1000_100_10_10_2d.csv")
# two <- read.csv("1000_300_10_10_2d.csv")
# three <- read.csv("1100_100_10_10_2d.csv")
# four <- read.csv("800_200_10_10_2d.csv")
# five <- read.csv("900_200_10_10_2d.csv")

# Load the SdA 20 -> 2d data 
# one <- read.csv("1000_400_20_20_2d.csv")
# two <- read.csv("1100_100_20_20_2d.csv")
# three <- read.csv("1100_300_20_20_2d.csv")
# four <- read.csv("800_200_20_20_2d.csv")
# five <- read.csv("900_300_20_20_2d.csv")

# Load the SdA 50 -> 2d data
one <- read.csv("1000_100_50_50_2d.csv")
two <- read.csv("1000_200_50_50_2d.csv")
three <- read.csv("1100_300_50_50_2d.csv")
four <- read.csv("800_400_50_50_2d.csv")
five <- read.csv("900_300_50_50_2d.csv")

# Combine together
sda_lle <- rbind(lle_2d,one)
sda_pca <- rbind(pca_2d,two)
sda_kpca <- rbind(kpca_2d,three)
sda_isomap <- rbind(isomap_2d,four)

# Scatterplots, coloured by class.
sda_vs_lle <- ggplot(sda_lle, aes(x=X, y=Y, colour=label)) + geom_point(shape=1,alpha=0.5)
sda_vs_lle <- sda_vs_lle + facet_wrap(~ algorithm)

sda_vs_pca <- ggplot(sda_pca, aes(x=X, y=Y, colour=label)) + geom_point(shape=1,alpha=0.5)
sda_vs_pca <- sda_vs_pca + facet_wrap(~ algorithm)

sda_vs_kpca <- ggplot(sda_kpca, aes(x=X, y=Y, colour=label)) + geom_point(shape=1,alpha=0.5)
sda_vs_kpca <- sda_vs_kpca + facet_wrap(~ algorithm)

sda_vs_isomap <- ggplot(sda_isomap, aes(x=X, y=Y, colour=label)) + geom_point(shape=1,alpha=0.5)
sda_vs_isomap <- sda_vs_isomap + facet_wrap(~ algorithm)