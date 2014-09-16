library(ggplot2)
library(dplyr)

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
comparators <- rbind(isomap_10, kpca_10, pca_10, lle_10)
rm(isomap_10,lle_10,pca_10,kpca_10)

# Load the SdA reduced distances data, rbind in one df
setwd("/data/sda_output_data/distances/csv_data/sda_3layers")
one <- read.csv("1000_100_10.csv")
two <- read.csv("1000_300_10.csv")
three <- read.csv("1100_100_10.csv")
four <- read.csv("800_200_10.csv")
five <- read.csv("900_200_10.csv")
sdas <- rbind(one,two,three,four,five)
rm(one,two,three,four,five)

distances <- rbind(subset(comparators, select = -dimension),subset(sda_distances, select = -dimension))

# Plot densities of the distances between pts of a similar label
gp <- ggplot(distances, aes(x=label, fill=algorithm)) + geom_density(alpha=.3)
gp

