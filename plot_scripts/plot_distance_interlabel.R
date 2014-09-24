library(ggplot2)
library(dplyr)
library(gridExtra)


# Load the comparators reduced distances data 
setwd("/data/sda_output_data/distances/csv_data/comparators")
isomap_10 <- read.csv("isomap_dim10_interlabel.csv")
kpca_10 <- read.csv("kpca_dim10_interlabel.csv")
pca_10 <- read.csv("pca_dim10_interlabel.csv")
lle_10 <- read.csv("lle_dim10_interlabel.csv")

# Load the SdA reduced distances data, rbind in one df
setwd("/data/sda_output_data/distances/csv_data/sda_3layers")
one <- read.csv("1000_100_10_interlabel.csv")
two <- read.csv("1000_300_10_interlabel.csv")
three <- read.csv("1100_100_10_interlabel.csv")
four <- read.csv("800_200_10_interlabel.csv")
five <- read.csv("900_200_10_interlabel.csv")

sda_lle <- rbind(lle_10,one)
sda_pca <- rbind(pca_10,two)
sda_kpca <- rbind(kpca_10,three)
sda_isomap <- rbind(isomap_10,four)


# box + whisker plots
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
pdf("sda_vs_comparators_inter_distances.pdf")
grid.arrange(sda_vs_lle, sda_vs_kpca, sda_vs_isomap, sda_vs_pca, nrow=4)
dev.off()


