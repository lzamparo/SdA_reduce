library(ggplot2)

# load the data
three_layers <- read.csv('three_layer_model.csv', colClasses=c('numeric','numeric','numeric','factor','factor'))
# ditch the first column, which is epochs numbered from zero.   
three_layers <- three_layers[,c("epoch","score","value","param")]

# Faceted 3-param plot
three_p <- ggplot(three_layers, aes(epoch,score, colour = value))
three_p <- three_p + geom_line(aes(group=value))
three_p<- three_p + scale_x_discrete(limits = c(0,300),breaks = seq(0,300,by=10))
three_p <- three_p + scale_y_continuous(limits = c(50,300))
three_p <- three_p + facet_wrap( ~ param)


# Single plot: Learning Rate plot
three_p_lr <- ggplot(subset(three_layers, param == "learning_rate"), aes(epoch,score,colour = value))
three_p_lr <- three_p_lr + geom_point(alpha = 1/5)
three_p_lr <- three_p_lr + scale_x_discrete(limits = c(0,300),breaks = seq(0,300,by=10))
three_p_lr <- three_p_lr + scale_y_continuous(limits = c(50,300))
three_p_lr <- three_p_lr + stat_smooth()
three_p_lr <- three_p_lr + labs(colour="Learning Rate") + opts(title = "Validation score vs epoch: Learning Rate")
three_p_lr

# Single plot: Weight-decay plot
three_p_wd <- ggplot(subset(three_layers, param == "weight_decay"), aes(epoch,score,colour = value))
three_p_wd <- three_p_wd + geom_point(alpha = 1/5)
three_p_wd <- three_p_wd + scale_x_discrete(limits = c(0,300),breaks = seq(0,300,by=10))
three_p_wd <- three_p_wd + scale_y_continuous(limits = c(50,300))
three_p_wd <- three_p_wd + stat_smooth()
three_p_wd <- three_p_wd + labs(colour="Weight Decay") + opts(title = "Validation score vs epoch: Weight Decay")
three_p_wd

# Single plot: Momentum plot
three_p_mom <- ggplot(subset(three_layers, param == "momentum"), aes(epoch,score,colour = value))
three_p_mom <- three_p_mom + geom_point(alpha = 1/5)
three_p_mom <- three_p_mom + scale_x_discrete(limits = c(0,300),breaks = seq(0,300,by=10))
three_p_mom <- three_p_mom + scale_y_continuous(limits = c(50,300))
three_p_mom <- three_p_mom + stat_smooth()
three_p_mom <- three_p_mom + labs(colour="Momentum") + opts(title = "Validation score vs epoch: Momentum")
three_p_mom