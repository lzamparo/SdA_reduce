library(ggplot2)

curr_dir <- getwd()
setwd('/data/sda_output_data/test_hyperparam_output')

# load the data, ditch the first column, which is epochs numbered from zero
three_layers <- read.csv('three_layer_model.csv', colClasses=c('numeric','numeric','numeric','factor','factor'))
three_layers <- three_layers[,c("epoch","score","value","param")]
levels(three_layers$param) <- c("Learning Rate", "Momentum", "Weight Decay")

four_layers <- read.csv('four_layer_model.csv', colClasses=c('numeric','numeric','numeric','factor','factor'))
four_layers <- four_layers[,c("epoch","score","value","param")]
levels(four_layers$param) <- c("Learning Rate", "Momentum", "Weight Decay")

both_layers <- read.csv('both_models.csv', colClasses=c('numeric','numeric','factor','factor','numeric','factor'))
both_layers <- both_layers[,c("epoch","model","score","value","param")]
levels(both_layers$param) <- c("Learning Rate", "Momentum", "Weight Decay")

# Faceted plot, 3-params 2 models
full_p <- ggplot(both_layers, aes(epoch,score, colour = value))
full_p <- full_p + geom_point(alpha = 1/4)
full_p <- full_p + scale_x_discrete(limits = c(0,300), breaks = seq(0,300,by=10))
full_p <- full_p + scale_y_continuous(limits = c(50,300))
full_p <- full_p + facet_grid(model ~ param)
full_p <- full_p + stat_smooth()
full_p <- full_p + theme(strip.text.x = element_text(size = 13, face = "bold"))
full_p

# Faceted 3-param plot for 3-layer model
three_p <- ggplot(three_layers, aes(epoch,score, colour = value))
three_p <- three_p + geom_line(aes(group=value))
three_p<- three_p + scale_x_discrete(limits = c(0,300),breaks = seq(0,300,by=10))
three_p <- three_p + scale_y_continuous(limits = c(50,300))
three_p <- three_p + facet_wrap( ~ param)
three_p <- three_p + theme(strip.text.x = element_text(size = 13, face = "bold"))
three_p

# Faceted 3-param plot for 4-layer model
four_p <- ggplot(four_layers, aes(epoch,score, colour = value))
four_p <- four_p + geom_line(aes(group=value))
four_p<- four_p + scale_x_discrete(limits = c(0,300),breaks = seq(0,300,by=10))
four_p <- four_p + scale_y_continuous(limits = c(50,300))
four_p <- four_p + facet_wrap( ~ param)
four_p <- four_p + theme(strip.text.x = element_text(size = 13, face = "bold"))
four_p

# Single plots: Learning Rate plot
full_p_lr <- ggplot(subset(both_layers, param == "Learning Rate"), aes(epoch,score,colour = value))
full_p_lr <- full_p_lr + geom_point(alpha = 1/4)
full_p_lr <- full_p_lr + scale_x_discrete(limits = c(0,300), breaks = seq(0,300,by=10))
full_p_lr <- full_p_lr + scale_y_continuous(limits = c(50,300))
full_p_lr <- full_p_lr + stat_smooth()
full_p_lr <- full_p_lr + ggtitle("Validation score vs epoch: Learning Rate")
full_p_lr

three_p_lr <- ggplot(subset(three_layers, param == "Learning Rate"), aes(epoch,score,colour = value))
three_p_lr <- three_p_lr + geom_point(alpha = 1/4)
three_p_lr <- three_p_lr + scale_x_discrete(limits = c(0,300),breaks = seq(0,300,by=10))
three_p_lr <- three_p_lr + scale_y_continuous(limits = c(50,300))
three_p_lr <- three_p_lr + stat_smooth()
three_p_lr <- three_p_lr + ggtitle("Validation score vs epoch: Learning Rate")
three_p_lr

four_p_lr <- ggplot(subset(four_layers, param == "Learning Rate"), aes(epoch,score,colour = value))
four_p_lr <- four_p_lr + geom_point(alpha = 1/4)
four_p_lr <- four_p_lr + scale_x_discrete(limits = c(0,300),breaks = seq(0,300,by=10))
four_p_lr <- four_p_lr + scale_y_continuous(limits = c(50,300))
four_p_lr <- four_p_lr + stat_smooth()
four_p_lr <- four_p_lr + ggtitle("Validation score vs epoch: Learning Rate")
four_p_lr

# Single plot: Weight-decay plot
three_p_wd <- ggplot(subset(three_layers, param == "Weight Decay"), aes(epoch,score,colour = value))
three_p_wd <- three_p_wd + geom_point(alpha = 1/4)
three_p_wd <- three_p_wd + scale_x_discrete(limits = c(0,300),breaks = seq(0,300,by=10))
three_p_wd <- three_p_wd + scale_y_continuous(limits = c(50,300))
three_p_wd <- three_p_wd + stat_smooth()
three_p_wd <- three_p_wd + ggtitle("Validation score vs epoch: Weight Decay")
three_p_wd

four_p_wd <- ggplot(subset(four_layers, param == "Weight Decay"), aes(epoch,score,colour = value))
four_p_wd <- four_p_wd + geom_point(alpha = 1/4)
four_p_wd <- four_p_wd + scale_x_discrete(limits = c(0,300),breaks = seq(0,300,by=10))
four_p_wd <- four_p_wd + scale_y_continuous(limits = c(50,300))
four_p_wd <- four_p_wd + stat_smooth()
four_p_wd <- four_p_wd + ggtitle("Validation score vs epoch: Weight Decay")
four_p_wd

# Single plot: Momentum plot
three_p_mom <- ggplot(subset(three_layers, param == "Momentum"), aes(epoch,score,colour = value))
three_p_mom <- three_p_mom + geom_point(alpha = 1/4)
three_p_mom <- three_p_mom + scale_x_discrete(limits = c(0,300),breaks = seq(0,300,by=10))
three_p_mom <- three_p_mom + scale_y_continuous(limits = c(50,300))
three_p_mom <- three_p_mom + stat_smooth()
three_p_mom <- three_p_mom + ggtitle("Validation score vs epoch: Momentum")
three_p_mom

four_p_mom <- ggplot(subset(four_layers, param == "Momentum"), aes(epoch,score,colour = value))
four_p_mom <- four_p_mom + geom_point(alpha = 1/4)
four_p_mom <- four_p_mom + scale_x_discrete(limits = c(0,300),breaks = seq(0,300,by=10))
four_p_mom <- four_p_mom + scale_y_continuous(limits = c(50,300))
four_p_mom <- four_p_mom + stat_smooth()
four_p_mom <- four_p_mom + ggtitle("Validation score vs epoch: Momentum")
four_p_mom