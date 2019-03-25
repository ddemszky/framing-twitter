library(ggplot2)
library(ggrepel)
library(RColorBrewer)
library(gridExtra)
library(grid)
library(scales)

setwd('/Users/ddemszky/Google_Drive/Research/Framing/NAACL/framing-twitter')
data <- read.csv("data/topic_eval/collapsed_tweet_results.csv",header=TRUE)
data2 <- read.csv("data/topic_eval/collapsed_word_results.csv",header=TRUE)

cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

p1 = ggplot(data=data, aes(x=reorder(model, accuracy), y=accuracy, fill=reorder(model, accuracy))) +
  geom_bar(stat="identity", ) +
  geom_errorbar(aes(ymin=accuracy-se, ymax=accuracy+se), width=.2) +
  xlab("") +
  ylab("Accuracy") +
  theme_bw(base_size=13) +
  ggtitle("Tweet intrusion") +
  scale_fill_manual(values=cbPalette) +
  guides(fill=FALSE) +
  scale_y_continuous(limits=c(.4,.8),oob = rescale_none)
  
p1

p2 = ggplot(data=data2, aes(x=reorder(model, accuracy), y=accuracy, fill=reorder(model, accuracy))) +
  geom_bar(stat="identity", ) +
  geom_errorbar(aes(ymin=accuracy-se, ymax=accuracy+se), width=.2) +
  xlab("") +
  ylab("") +
  theme_bw(base_size=13) +
  ggtitle("Word intrusion")+
  scale_fill_manual(values=cbPalette) +
  guides(fill=FALSE) +
  scale_y_continuous(limits=c(.2,.55),oob = rescale_none)

grid.arrange(p1, p2, nrow=1)
