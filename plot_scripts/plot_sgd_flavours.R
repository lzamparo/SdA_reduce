library(ggplot2)

curr_dir <- getwd()
setwd('/data/sda_output_data/sgd_flavours')
both_layers <- read.csv('both_models.csv')

levels(both_layers$group) <- c("3 Layers","4 Layers")
levels(both_layers$method) <- c("Adagrad", "Adagrad WD","CM","NAG")

# curtail all data at 50 epochs
curtailed_data <- subset(both_layers, epoch <= 50)

curtailed_p <- ggplot(curtailed_data, aes(epoch,score, colour = method))
curtailed_p <- curtailed_p + geom_smooth(alpha = 1/6)
curtailed_p <- curtailed_p + scale_x_discrete(limits = c(1,50), breaks = seq(0,50,by=10))
curtailed_p <- curtailed_p + scale_y_continuous(limits = c(100,700))
curtailed_p <- curtailed_p + facet_wrap( ~ group)
#curtailed_p <- curtailed_p + stat_smooth()
curtailed_p <- curtailed_p + theme(strip.text.x = element_text(size = 13))
curtailed_p <- curtailed_p + theme(axis.text.x = element_text(size = 13, face = "bold"))
curtailed_p <- curtailed_p + theme(axis.title.x = element_text(size = 13, face = "bold"))
curtailed_p <- curtailed_p + theme(axis.text.y = element_text(size = 13, face = "bold"))
curtailed_p <- curtailed_p + theme(axis.title.y = element_text(size = 13, face = "bold"))
curtailed_p <- curtailed_p + theme(legend.text = element_text(size = 13))
curtailed_p <- curtailed_p + theme(legend.title = element_text(size = 13))
curtailed_p <- curtailed_p + labs(colour = "SGD Method")
curtailed_p <- curtailed_p + ggtitle("Validation score vs epoch: optimization algorithms")
curtailed_p <- curtailed_p + theme(plot.title = element_text(size=15, face = "bold"))
curtailed_p

just_adagrad <- subset(both_layers, method == 'Adagrad' | method == 'Adagrad WD')
adagrad_p <- ggplot(just_adagrad, aes(epoch, score, colour = method))
adagrad_p <- adagrad_p + geom_dodge()
adagrad_p <- adagrad_p + scale_x_discrete(limits = c(1,300), breaks = seq(0,300,by=10))
adagrad_p <- adagrad_p + scale_y_continuous(limits = c(100,450))
adagrad_p <- adagrad_p + facet_wrap( ~ group)
adagrad_p <- adagrad_p + stat_smooth(size = 2)
adagrad_p <- adagrad_p + labs(colour = "SGD Method")
adagrad_p <- adagrad_p + ggtitle("Validation score vs epoch: adagrad with or without weight decay")
adagrad_p

