library(ggplot2)
library(dplyr)

# Load the SdA models data
setwd("/data/sda_output_data/homogeneity")
sda_df <- read.csv('all_sda_models.csv')

# find top performing 3 layer, 4 layer SdA models
three_layers <- sda_df %>% filter(Layers == "3_layers") %>% group_by(Dimension) %>% filter(min_rank(desc(Homogeneity)) < 6)
four_layers <- sda_df %>% filter(Layers == "4_layers") %>% group_by(Dimension) %>% filter(min_rank(desc(Homogeneity)) < 6)

# Load the comparators reduced statistics data (not really a good description, but I have to go so it will have to do)
setwd("/data/sda_output_data/homogeneity/csv_data/dfs")
comparators_m <- read.csv("comparators_mahalanobis.csv")
comparators_e <- read.csv("comparators_euclidean.csv")

dim10 <- comparators_e %>% filter(dimension == "dim10") %>% group_by(algorithm,label)
dim10_samples <- comparators_e %>% filter(dimension == "dim10") %>% group_by(algorithm,label) %>% rowwise() %>% do(samples = rnorm(10,mean = .$mean, sd = sqrt(.$var)))

# Breaking from Hadley Wickham: use rowwise() and do() at the end of the dplyr chain to generate samples w rnorm. 
