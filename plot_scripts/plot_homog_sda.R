library(ggplot2)
library(dplyr)

# Load the data
setwd("/data/sda_output_data/homogeneity")

comparators_df <- read.csv("all_comparator_models.csv")
levels(comparators_df$Model) <- c("Isomap","k-PCA","LLE","PCA")

# get the SdA data 
sda_df <- read.csv('all_sda_models.csv')
# separate 3 layer, 4 layer top 5 models
three_layers <- sda_df %>% filter(Layers == "3_layers") %>% group_by(Dimension) %>% filter(min_rank(desc(Homogeneity)) < 6)
four_layers <- sda_df %>% filter(Layers == "4_layers") %>% group_by(Dimension) %>% filter(min_rank(desc(Homogeneity)) < 6)
# re-label
three_layers_homog <- three_layers[,c("Dimension","Homogeneity","Model")]
levels(three_layers_homog$Model) <- rep("SdA 3 layers", length(levels(three_layers_homog$Model)))
four_layers_homog <- four_layers[,c("Dimension","Homogeneity","Model")]
levels(four_layers_homog$Model) <- rep("SdA 4 layers", length(levels(four_layers_homog$Model)))

master_df <- rbind(comparators_df,three_layers_homog,four_layers_homog)

# Plot the data
homog <- ggplot(master_df, aes(Dimension,Homogeneity, colour = Model))

# Set the colour palette
homog <- homog + scale_colour_brewer(palette="Set1")

# Use the manual colour-blind friendly palette
#cbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
#homog <- homog + scale_colour_manual(values=cbPalette)

# Chang the background theme
#homog <- homog + theme_bw()

# Plot the points, trend-line, error-ribbons
homog <- homog + geom_point(alpha = 1/4)
homog <- homog + stat_smooth(size=1.5,alpha=1/2)
homog <- homog + labs(colour = "Model")
#homog <- homog + ggtitle("Average Homogeneity vs Dimension")
#homog <- homog + theme(plot.title = element_text(size=15, face = "bold"))
homog <- homog + theme(strip.text.x = element_text(size = 13))
homog <- homog + theme(legend.title = element_text(size = 13))
homog <- homog + theme(legend.text = element_text(size = 13))
homog <- homog + theme(axis.title.y = element_text(size = 15))
homog <- homog + theme(axis.text.y = element_text(size = 13))
homog <- homog + theme(axis.title.x = element_text(size = 15))
homog <- homog + theme(axis.text.x = element_text(size = 13))
homog