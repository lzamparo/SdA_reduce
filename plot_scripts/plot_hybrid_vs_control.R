library(ggplot2)

curr_dir <- getwd()
setwd('/data/sda_output_data/pretraining_hybrid_vs_control')

# load the data, ditch the first column, which is epochs numbered from zero
both_methods <- read.csv('both_methods.csv', colClasses=c('factor','numeric','factor','factor','factor','factor','numeric'))
colnames(both_methods) <- c("dimension","epoch","layers","units","method","model","score")
levels(both_methods$layers) <- c("3", "4")

# Faceted plot: dimension x units
three_layers <- subset(both_methods, layers == '3')
four_layers <- subset(both_methods, layers == '4')

three_fig <- ggplot(three_layers, aes(epoch,score, colour = method))
three_fig <- three_fig + geom_line(alpha = 1/9, size = 2)
three_fig <- three_fig + scale_x_discrete(limits = c(0,300), breaks = seq(0,300,by=10))
three_fig <- three_fig + scale_y_continuous(limits = c(50,1500))
three_fig <- three_fig + facet_grid(dimension ~ units) + theme(panel.margin = unit(0.75, "lines"))
three_fig <- three_fig + stat_smooth()
three_fig <- three_fig + theme(strip.text.x = element_text(size = 13, face = "bold"))
three_fig <- three_fig + theme(strip.text.y = element_text(size = 13, face = "bold"))
three_fig <- three_fig + theme(legend.title = element_text(size = 13))
three_fig <- three_fig + theme(legend.text = element_text(size = 13))
three_fig <- three_fig + labs(colour = "Pre-training")
three_fig <- three_fig + ggtitle("Comparing Pre-training Methods: Activation function vs Dimensions") + theme(plot.title=element_text(family="Times", face="bold", size=15))
three_fig
  
# The 4 layer data is more variable in the ReLU units than the 3 layer data, this is here for completeness' sake.

four_fig <- ggplot(four_layers, aes(epoch,score, colour = method))
four_fig <- four_fig + geom_line(alpha = 1/8,size = 2)
four_fig <- four_fig + scale_x_discrete(limits = c(0,300), breaks = seq(0,300,by=10))
four_fig <- four_fig + scale_y_continuous(limits = c(50,1500))
four_fig <- four_fig + facet_grid(dimension ~ units)
four_fig <- four_fig + stat_smooth()
four_fig <- four_fig + theme(strip.text.x = element_text(size = 13, face = "bold"))
four_fig <- four_fig + theme(strip.text.y = element_text(size = 13, face = "bold"))
four_fig <- four_fig + theme(legend.title = element_text(size = 13))
four_fig <- four_fig + theme(legend.text = element_text(size = 13))
four_fig <- four_fig + labs(colour = "Pre-training")
four_fig <- four_fig + ggtitle("Comparing Pre-training Methods: Activation vs Dimensions")

# There's a *ton* of overplotting that takes place here; too much data!

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
