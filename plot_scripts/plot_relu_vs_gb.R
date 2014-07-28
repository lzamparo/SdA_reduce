library(ggplot2)

curr_dir <- getwd()
setwd('/data/sda_output_data/relu_vs_gb')
both_layers <- read.csv('both_models.csv')

levels(both_layers$group) <- c("3 layers", "5 layers")

# Faceted plot, 3-params 2 models
full_p <- ggplot(both_layers, aes(epoch,score, colour = layer))
full_p <- full_p + geom_point(alpha = 1/4)
full_p <- full_p + scale_x_discrete(limits = c(1,50), breaks = seq(0,50,by=10))
full_p <- full_p + scale_y_continuous(limits = c(100,700))
full_p <- full_p + facet_wrap( ~ group)
full_p <- full_p + stat_smooth()
full_p <- full_p + theme(strip.text.x = element_text(size = 13))
full_p <- full_p + theme(axis.text.x = element_text(size = 13, face = "bold"))
full_p <- full_p + theme(axis.title.x = element_text(size = 13, face = "bold"))
full_p <- full_p + theme(axis.text.y = element_text(size = 13, face = "bold"))
full_p <- full_p + theme(axis.title.y = element_text(size = 13, face = "bold"))
full_p <- full_p + theme(legend.text = element_text(size = 13))
full_p <- full_p + theme(legend.title = element_text(size = 13))
full_p <- full_p + labs(colour = "Units")
full_p <- full_p + ggtitle("Validation score vs epoch: ReLU units vs Gassian+Bernoulli units")
full_p <- full_p + theme(plot.title = element_text(size=15, face = "bold"))
full_p