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

model.c <- lmer(leaveout ~ 1 + (1|event), data = data)
model.a <- lmer(leaveout ~ 1 + time + (1|event), data = data)
anova(model.a, model.c)

model.a %>% summary() %>% print()

model.c <- lmer(leaveout ~ 1 + (1|event) + (0 + time|event), data = data)
model.a <- lmer(leaveout ~ 1 + time + (1|event) + (0 + time|event), data = data)
anova(model.a, model.c)
model.a %>% summary() %>% print()
model.a %>% coef()

model.c <- lmer(leaveout ~ 1 + (1 + time|event), data = data)
model.a <- lmer(leaveout ~ 1 + time + (1 + time|event), data = data)
anova(model.a, model.c)
model.a %>% summary() %>% print()
model.a %>% coef()
data %>% 
  filter(event == 'roseburg') %>% 
  lm(leaveout ~ time, data=.) %>% 
  summary()
  




ggplot(data, aes(x=time, y=leaveout)) + 
  geom_point() + 
  geom_smooth(method ="lm",
              se=TRUE) +
  facet_wrap(~event)

ggplot(data, aes(x=log_time, y=leaveout)) + 
  geom_point() + 
  geom_smooth(method ="lm",
              se=TRUE,
              level =.99)

# degree 2 polynomial
#ggplot(data, aes(x=time, y=leaveout)) + 
#  geom_point() +  
#  geom_smooth(method ="lm",
#              formula = y ~ x + I(x^2),
#              se=TRUE) +
#  facet_wrap(~event)

