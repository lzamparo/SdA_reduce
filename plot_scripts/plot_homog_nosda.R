library(ggplot2)

# Load the data
setwd("/data/sda_output_data/homogeneity")
all_comparators <- read.csv("all_comparator_models.csv")


# Plot the data
homog <- ggplot(all_comparators, aes(Dimension,Homogeneity, colour = Model))

# Set the colour palette
homog <- homog + scale_colour_brewer(palette="Set1")

homog <- homog + geom_point(alpha = 1/5)
homog <- homog + stat_smooth(size=1.5,alpha=1/2)
# homog <- homog + scale_x_reverse()
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