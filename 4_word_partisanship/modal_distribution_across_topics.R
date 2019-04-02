library(ggplot2)
library(ggrepel)
library(RColorBrewer)
setwd('/Users/ddemszky/Google_Drive/Research/Framing/NAACL/framing-twitter/data/output')
data <- read.csv("modal_distributions.csv",header=TRUE)
meaned <- aggregate(proportion ~ (topic + modal), data, mean)
g <- ggplot(meaned, aes(x=reorder(topic, proportion), y=proportion, fill=modal))
g + geom_bar(stat='identity', position='dodge', width = 0.7, color='black') + 
  theme_bw(base_size=14) +
  theme(axis.text.y = element_text(size=12.5), legend.position="top", legend.direction="horizontal", legend.title=element_text(size=12), legend.margin=margin(0,0,0,0),legend.box.margin=margin(-1,-10,-10,-10)) +
  ylab('Percentage of modal in topic / overall topic percentage') +
  xlab('') +
  scale_fill_manual(values = rev(brewer.pal(5,"BuPu"))) +
  guides(fill=guide_legend(reverse = TRUE, title="Modal:", size=14)) +
  coord_flip()



data <- read.csv("pronoun_distributions.csv",header=TRUE)
meaned <- aggregate(proportion ~ (topic + pronoun), data, mean)
#meaned$pronoun <- factor(c("I","SheHe","They", "We", 'You'),levels=rev(c("I","We", 'You', "They", "SheHe")))
#meaned$topic <- factor(c("ideology","investigation","laws &\npolicy", 'news', 'remembrance', 'solidarity'),levels=c("investigation", "ideology",'news', 'remembrance', "laws &\npolicy", 'solidarity'))

g <- ggplot(meaned, aes(x=reorder(topic, proportion), y=proportion, fill=pronoun))
g + geom_bar(stat='identity', position='dodge', width = 0.7, color='black') + 
  theme_bw(base_size=14) +
  theme(axis.text.y = element_text(size=12.5), legend.position="top", legend.direction="horizontal", legend.title=element_text(size=13), legend.margin=margin(0,0,0,0),legend.box.margin=margin(-1,-10,-10,-10)) +
  ylab('Percentage of pronoun in topic / overall topic percentage') +
  xlab('') +scale_fill_brewer(palette="RdYlGn") +
  guides(fill=guide_legend(reverse = FALSE, title="Pronoun:", size=14)) +
  coord_flip()
