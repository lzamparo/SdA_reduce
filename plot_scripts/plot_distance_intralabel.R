library(ggplot2)
library(dplyr)
library(gridExtra)

# Load the SdA models data
setwd("/data/sda_output_data/homogeneity")
sda_df <- read.csv('all_sda_models.csv')

# find top performing 3 layer, 4 layer SdA models
three_layers <- sda_df %>% filter(Layers == "3_layers") %>% group_by(Dimension) %>% filter(min_rank(desc(Homogeneity)) < 6)
four_layers <- sda_df %>% filter(Layers == "4_layers") %>% group_by(Dimension) %>% filter(min_rank(desc(Homogeneity)) < 6)

# Load the comparators reduced distances data 
setwd("/data/sda_output_data/distances/csv_data/comparators")
isomap_10 <- read.csv("isomap_dim10.csv")
kpca_10 <- read.csv("kpca_dim10.csv")
pca_10 <- read.csv("pca_dim10.csv")
lle_10 <- read.csv("lle_dim10.csv")

# Load the SdA reduced distances data, rbind in one df
setwd("/data/sda_output_data/distances/csv_data/sda_3layers")
one <- read.csv("1000_100_10.csv")
two <- read.csv("1000_300_10.csv")
three <- read.csv("1100_100_10.csv")
four <- read.csv("800_200_10.csv")
five <- read.csv("900_200_10.csv")

sda_lle <- rbind(subset(lle_10, select = -dimension),subset(one, select = -dimension))
sda_pca <- rbind(subset(pca_10, select = -dimension),subset(two, select = -dimension))
sda_kpca <- rbind(subset(kpca_10, select = -dimension),subset(three, select = -dimension))
sda_isomap <- rbind(subset(isomap_10, select = -dimension),subset(four, select = -dimension))



# I'd like to plot densities of the distances between pts of a similar label
# This is complicated by the sheer number of points involved, as well as the 
# different scales of distances

# Second try: a grid of 3 separate 3-facet_wrap plots: each facet is SdA versus <comparator> in <label>?

# sda_vs_lle <- ggplot(sda_lle, aes(x=distances, fill=algorithm)) + geom_density(alpha=.5)
# sda_vs_lle <- sda_vs_lle + facet_wrap(~ label)
# 
# sda_vs_kpca <- ggplot(sda_kpca, aes(x=distances, fill=algorithm)) + geom_density(alpha=.5)
# sda_vs_kpca <- sda_vs_kpca + facet_wrap(~ label)
# 
# sda_vs_isomap <- ggplot(sda_isomap, aes(x=distances, fill=algorithm)) + geom_density(alpha=.5)
# sda_vs_isomap <- sda_vs_isomap + facet_wrap(~ label)
# 
# sda_vs_pca <- ggplot(sda_pca, aes(x=distances, fill=algorithm)) + geom_density(alpha=.5)
# sda_vs_pca <- sda_vs_pca + facet_wrap(~ label)

# Third try: box + whisker plots
sda_vs_lle <- ggplot(sda_lle, aes(x=algorithm, y=distances, fill=algorithm)) + geom_boxplot() +
  guides(fill=FALSE)
sda_vs_lle <- sda_vs_lle + facet_wrap(~ label)

sda_vs_pca <- ggplot(sda_pca, aes(x=algorithm, y=distances, fill=algorithm)) + geom_boxplot() +
  guides(fill=FALSE)
sda_vs_pca <- sda_vs_pca + facet_wrap(~ label)

sda_vs_isomap <- ggplot(sda_isomap, aes(x=algorithm, y=distances, fill=algorithm)) + geom_boxplot() +
  guides(fill=FALSE)
sda_vs_isomap <- sda_vs_isomap + facet_wrap(~ label)

sda_vs_kpca <- ggplot(sda_kpca, aes(x=algorithm, y=distances, fill=algorithm)) + geom_boxplot() +
  guides(fill=FALSE)
sda_vs_kpca <- sda_vs_kpca + facet_wrap(~ label)

# stack these, save to file. 
pdf("sda_vs_comparators_intra_distances.pdf")
grid.arrange(sda_vs_lle, sda_vs_kpca, sda_vs_isomap, sda_vs_pca, nrow=4)
dev.off()



