library(ggplot2)
library(ggrepel)
library(RColorBrewer)
library("knitr")      # for knitting RMarkdown 
library("tidyverse")  # for wrangling, plotting, etc. 
library("lme4")

setwd('/Users/ddemszky/Google_Drive/Research/Framing/NAACL/framing-twitter/data/output')
data <- read.csv("polarization_over_time_log.csv",header=TRUE)

day = 60 * 60 * 24
data <- data %>% 
  filter(event != 'fort_lauderdale',
         leaveout != 0.5,
         leaveout > .49,
         squared_diff < 0.01) %>% 
  mutate(log_time = log(time * day))


ggplot(data, aes(x=log_time, y=leaveout)) + 
  geom_point() + 
  geom_smooth(method ="lm",
              se=TRUE,
              level = .90) +
  facet_wrap(~event) 

ggplot(data, aes(x=log_time, y=leaveout)) + 
  geom_point() + 
  geom_smooth(method ="lm",
              se=TRUE,
              level =.99)

ggplot(data, aes(x=time, y=leaveout)) + 
  geom_point() +  
  geom_smooth(method ="lm",
              formula = y ~ x + I(x^2),
              se=TRUE) +
  facet_wrap(~event)

model.c <- lmer(leaveout ~ 1 + (1 + time|event), data = data)
model.a <- lmer(leaveout ~ 1 + time + (1 + time|event), data = data)
#model.q <- lmer(leaveout ~ time + I(time^2) + (time|event), data = data)
summary(model.a)

anova(model.a, model.c)

model.basic <- lm(leaveout ~ log_time, data = data)
summary(model.basic)
