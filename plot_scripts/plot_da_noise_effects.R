library(ggplot2)
library(ggthemes)

curr_dir <- getwd()
setwd('/data/sda_output_data/')
noise_df <- read.csv("/data/dA_results/noise_df.csv", colClasses=c("factor","factor","integer","numeric"))

noise_df_totwenty <- subset(noise_df, Epoch <= 20)

# Faceted plot, 2 activation types
full_p <- ggplot(noise_df_totwenty, aes(Epoch,Value, colour = Corruption))
full_p <- full_p + geom_line() + scale_color_grey()
full_p <- full_p + facet_wrap( ~ Activation)
full_p <- full_p + ggtitle("Reconstruction Error vs epoch: corruption parameter values")
full_p <- full_p + theme(plot.title = element_text(size=15, face = "bold"))
full_p <- full_p + theme(strip.text.x = element_text(size = 13))
full_p <- full_p + theme(axis.text.x = element_text(size = 13, face = "bold"))
full_p <- full_p + theme(axis.title.x = element_text(size = 13, face = "bold"))
full_p <- full_p + theme(axis.text.y = element_text(size = 13, face = "bold"))
full_p <- full_p + theme(axis.title.y = element_text(size = 13, face = "bold"))
full_p <- full_p + theme(legend.text = element_text(size = 13))
full_p <- full_p + theme(legend.title = element_text(size = 13))
full_p



