library(ggplot2)
library(dplyr)

# Load the data
setwd("/data/sda_output_data/homogeneity")
isomap_df <- read.csv("isomap_df.csv")
kpca_df <- read.csv('kpca_df.csv')
pca_df <- read.csv('pca_df.csv')
lle_df <- read.csv('lle_df.csv')
master_df <- rbind(isomap_df,kpca_df,pca_df,lle_df)
levels(master_df$method) <- c("Isomap","k-PCA","PCA","LLE")
colnames(master_df) <- c("Homogeneity","Dimension","Model")

# get the SdA data 
sda_df <- read.csv('all_sda_models.csv')
# separate 3 layer, 4 layer top 5 models
three_layers <- sda_df %>% filter(Layers == "3_layers") %>% group_by(Dimension) %>% filter(min_rank(desc(Homogeneity)) < 6)
four_layers <- sda_df %>% filter(Layers == "4_layers") %>% group_by(Dimension) %>% filter(min_rank(desc(Homogeneity)) < 6)
# re-label
three_layers_homog <- three_layers[,c("Dimension","Homogeneity","Model")]
levels(three_layers_homog$Model) <- rep("SdA 3", length(levels(three_layers_homog$Model)))
four_layers_homog <- four_layers[,c("Dimension","Homogeneity","Model")]
levels(four_layers_homog$Model) <- rep("SdA 4", length(levels(four_layers_homog$Model)))

master_df <- rbind(master_df,three_layers_homog,four_layers_homog)

# Plot the data
homog <- ggplot(master_df, aes(Dimension,Homogeneity, colour = Model))
homog <- homog + geom_point(alpha = 1/4)
homog <- homog + stat_smooth()
homog <- homog + scale_x_reverse()
homog <- homog + labs(colour = "Model")
homog <- homog + ggtitle("Average Homogeneity vs Dimension")
homog <- homog + theme(plot.title = element_text(size=15, face = "bold"))
homog <- homog + theme(strip.text.x = element_text(size = 13))
homog <- homog + theme(legend.title = element_text(size = 13))
homog <- homog + theme(legend.text = element_text(size = 13))
#homog <- homog + theme(axis.title.y = element_text(size = 13, face = "bold"))
#homog <- homog + theme(axis.text.y = element_text(size = 13, face = "bold"))
#homog <- homog + theme(axis.title.x = element_text(size = 13, face = "bold"))
#homog <- homog + theme(axis.text.x = element_text(size = 13, face = "bold"))
homog