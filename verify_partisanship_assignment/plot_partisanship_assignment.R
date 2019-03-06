library(ggplot2)
library(ggrepel)
data <- read.csv("output/verify_party_assignment.csv", header=TRUE)

# We exclude DC because it has a disproportionately large number of news media Twitter handles that make it an outlier. Moreover, DC is not an official state.
data = data[data$abbr != 'DC',]

data$avg_part <- rowMeans(subset(data, select = c(rep_share, rep_twitter)), na.rm = TRUE)

ggplot(data) + 
  geom_smooth(method='lm', aes(x=rep_share, y=rep_twitter, weight=partisan_twitter), size=.1, color='black', se=TRUE, fullrange=TRUE)  + 
  geom_point(aes(x=rep_share, y=rep_twitter, size=partisan_twitter, color=avg_part)) + 
  geom_point(aes(x=rep_share, y=rep_twitter, size=partisan_twitter), shape = 1,colour = "black") +
  geom_text_repel(aes(x=rep_share, y=rep_twitter, label=abbr), size=3.7) + 
  scale_color_distiller("Avg partisanship", palette = "RdBu", limits=c(0,1)) + 
  theme_bw() +
  labs(x='Rep share in 2016 elections', y='Proportion of Rep users in our data', title='', size='No. of partisan\nusers') + 
  scale_y_continuous(breaks=seq(0,1,.1)) + scale_x_continuous(breaks=seq(0,1,.1), expand=c(0,0), limits=c(0.3, .77))  +
  theme(legend.background = element_rect(color="black",size = 0.1),legend.margin=margin(c(4,4,5,4)), legend.title=element_text(size=8))


fit <- lm(rep_share ~ rep_twitter, weights = partisan_twitter, data = data)
summary(fit)
