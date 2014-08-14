library(ggplot2)

# Load the data
setwd("/data/sda_output_data/homogeneity")
isomap_df <- read.csv("isomap_df.csv")
kpca_df <- read.csv('kpca_df.csv')
pca_df <- read.csv('pca_df.csv')
lle_df <- read.csv('lle_df.csv')
master_df <- rbind(isomap_df,kpca_df,pca_df,lle_df)
levels(master_df$method) <- c("Isomap","k-PCA","PCA","LLE")
colnames(master_df) <- c("Homogeneity","Dimension","Method")

# Plot the data
homog <- ggplot(master_df, aes(Dimension,Homogeneity, colour = Method))
homog <- homog + geom_point(alpha = 1/5)
homog <- homog + stat_smooth()
homog <- homog + scale_x_reverse()
homog <- homog + labs(colour = "Method")
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