library(ggplot2)
library(ggrepel)
library(RColorBrewer)

setwd('.')
data <- read.csv("data/output/modal_distributions.csv",header=TRUE)
meaned <- aggregate(proportion ~ (topic + modal), data, mean)
g <- ggplot(meaned, aes(x=reorder(topic, proportion), y=proportion, fill=modal))
g + geom_bar(stat='identity', position='dodge', width = 0.7, color='black') + 
  theme_bw(base_size=14) +
  theme(axis.text.y = element_text(size=14), legend.position="top", legend.direction="horizontal", legend.title=element_text(size=11), legend.margin=margin(0,0,0,0),legend.box.margin=margin(-1,-10,-10,-10)) +
  ylab('Percentage of modal in topic / overall topic percentage') +
  xlab('') +
  scale_fill_manual(values = rev(brewer.pal(5,"BuPu"))) +
  guides(fill=guide_legend(reverse = TRUE, title="Modal:")) +
  coord_flip()
