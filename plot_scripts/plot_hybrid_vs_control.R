library(ggplot2)

curr_dir <- getwd()
setwd('/data/sda_output_data/pretraining_hybrid_vs_control')

# load the data, ditch the first column, which is epochs numbered from zero
both_methods <- read.csv('both_methods.csv', colClasses=c('factor','numeric','factor','factor','factor','factor','numeric'))
colnames(both_methods) <- c("dimension","epoch","layers","units","method","model","score")
levels(both_methods$layers) <- c("3", "4")

# Faceted plot: dimension x units
full_p <- ggplot(both_methods, aes(epoch,score, colour = method))
full_p <- full_p + geom_line(alpha = 1/8,size = 2)
full_p <- full_p + scale_x_discrete(limits = c(0,300), breaks = seq(0,300,by=10))
full_p <- full_p + scale_y_continuous(limits = c(50,1500))
full_p <- full_p + facet_grid(dimension ~ units)
full_p <- full_p + stat_smooth()
full_p <- full_p + theme(strip.text.x = element_text(size = 13, face = "bold"))
full_p <- full_p + theme(strip.text.y = element_text(size = 13, face = "bold"))
full_p <- full_p + theme(legend.title = element_text(size = 13))
full_p <- full_p + theme(legend.text = element_text(size = 13))
full_p <- full_p + labs(colour = "Pre-training")
full_p <- full_p + ggtitle("Comparing Pre-training Methods: Activation vs Dimensions")
full_p