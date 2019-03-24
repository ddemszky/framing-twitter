library(ggplot2)
library(ggrepel)
library(RColorBrewer)
library("knitr")      # for knitting RMarkdown 
library("tidyverse")  # for wrangling, plotting, etc. 
library("lme4")

setwd('/Users/ddemszky/Google_Drive/Research/Framing/NAACL/framing-twitter/data/output')
data <- read.csv("polarization_over_time.csv",header=TRUE)
data2 <- read.csv("polarization_over_time_nomulti.csv",header=TRUE)

comb = merge(data, data2, c("event", "time"))

day = 60 * 60 * 24
data <- data %>% 
  filter(event != 'fort_lauderdale',
         leaveout != 0.5,
         leaveout > .49,
         squared_diff < 0.01) %>% 
  mutate(log_time = log(time * day))

data2 <- data2 %>% 
  filter(event != 'fort_lauderdale',
         leaveout != 0.5,
         leaveout > .49,
         squared_diff < 0.01) %>% 
  mutate(log_time = log(time * day))

comb <- comb %>% 
  filter(event != 'fort_lauderdale',
         leaveout.y != 0.5,
         leaveout.y > .49,
         leaveout.x > .49,
         squared_diff.y < 0.01,
         squared_diff.x < 0.01) %>% 
  mutate(log_time = log(time * day))


model.c <- lmer(leaveout ~ 1 + (1|event), data = data2)
model.a <- lmer(leaveout ~ 1 + time + (1|event), data = data2)
anova(model.a, model.c)
model.a %>% summary() %>% print()

model.c <- lmer(leaveout ~ 1 + (1|event) + (0 + time|event), data = data2)
model.a <- lmer(leaveout ~ 1 + time + (1|event) + (0 + time|event), data = data2)
anova(model.a, model.c)

model.c <- lmer(leaveout ~ 1 + (1 + time|event), data = data2)
model.a <- lmer(leaveout ~ 1 + time + (1 + time|event), data = data)
anova(model.a, model.c)

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
  geom_smooth(method ="lm",
              se=TRUE) +
  geom_point(aes(fill=event), size=3, shape=21, alpha=.7) +
  theme_bw() +
  xlab("Day after event") +
  ylab("Leave-out estimate") +
  guides(fill=guide_legend(ncol=3)) +
  theme(legend.position = c(0.25, 0.77), legend.background = element_rect(color = "black", "white", size = .5, linetype = "solid"),legend.title=element_blank())


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

