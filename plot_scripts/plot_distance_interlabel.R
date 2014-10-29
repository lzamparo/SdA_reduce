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

sda_lle <- rbind(subset(lle_10, select = -dimension),subset(one, select = -dimension))
sda_pca <- rbind(subset(pca_10, select = -dimension),subset(two, select = -dimension))
sda_kpca <- rbind(subset(kpca_10, select = -dimension),subset(three, select = -dimension))
sda_isomap <- rbind(subset(isomap_10, select = -dimension),subset(four, select= -dimension))

# Change the factor labels for opposing.label
levels(sda_lle$opposing.label) <- c("WT versus Foci", "WT versus Non-Round nuclei")
levels(sda_pca$opposing.label) <- c("WT versus Foci", "WT versus Non-Round nuclei")
levels(sda_kpca$opposing.label) <- c("WT versus Foci", "WT versus Non-Round nuclei")
levels(sda_isomap$opposing.label) <- c("WT versus Foci", "WT versus Non-Round nuclei")

# Use a different colour palette
cbPalette <- c("#E69F00", "#F0E442", "#000000", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7")

# box + whisker plots
sda_vs_lle <- ggplot(sda_lle, aes(x=algorithm, y=distances, fill=algorithm)) + geom_boxplot() +
  guides(fill=FALSE) + scale_fill_manual(values=cbPalette)
sda_vs_lle <- sda_vs_lle + facet_wrap(~ opposing.label)

sda_vs_pca <- ggplot(sda_pca, aes(x=algorithm, y=distances, fill=algorithm)) + geom_boxplot() +
  guides(fill=FALSE) + scale_fill_manual(values=cbPalette)
sda_vs_pca <- sda_vs_pca + facet_wrap(~ opposing.label)

sda_vs_isomap <- ggplot(sda_isomap, aes(x=algorithm, y=distances, fill=algorithm)) + geom_boxplot() +
  guides(fill=FALSE) + scale_fill_manual(values=cbPalette)
sda_vs_isomap <- sda_vs_isomap + facet_wrap(~ opposing.label)

sda_vs_kpca <- ggplot(sda_kpca, aes(x=algorithm, y=distances, fill=algorithm)) + geom_boxplot() +
  guides(fill=FALSE) 

# set the colour palette for the fill
sda_vs_kpca <- sda_vs_kpca + scale_fill_brewer(palette="Set1")
#sda_vs_kpca <- sda_vs_kpca + scale_fill_manual(values=cbPalette)

sda_vs_kpca <- sda_vs_kpca + facet_wrap(~ opposing.label)
sda_vs_kpca <- sda_vs_kpca + theme(strip.text.x = element_text(size = 13))
sda_vs_kpca <- sda_vs_kpca + theme(axis.text = element_text(size = 13))
sda_vs_kpca <- sda_vs_kpca + theme(axis.title = element_text(size = 13))

# Here's a nice density plot if needed
sda_vs_kpca <- ggplot(sda_kpca, aes(x=distances, fill=algorithm)) + geom_density(alpha = 0.3) + scale_colour_manual(values=cbPalette)
sda_vs_kpca <- sda_vs_kpca + facet_wrap(~ opposing.label)

# stack these, save to file. 
setwd("~/meetings/committee_meetings/sda_paper/images")
pdf("sda_vs_comparators_inter_distances.pdf")
grid.arrange(sda_vs_lle, sda_vs_kpca, sda_vs_isomap, sda_vs_pca, nrow=4)
dev.off()


