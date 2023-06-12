# P(B or M -> H) = P(B->H) + P(M -> H) - P(B and M -> H)
# headache: control 12/36 with headache, treatment 30/36 with headache
# headache ~ treatment_group, difference in probability = 18/36 (50%)
# 30/36 = 12/36 + 18/36 - P(B and M -> H)
# 30/36 = 12/36 + 27/36 - 9/36
# predict(glm, type="response") gives predicted probabilities from inverse logit transformation
# mean(predicted_prob_treat) - mean(predicted_prob_control)

# P(B or M -> H) = P(B->H) + P(M -> H) - 0
# P(B or M -> H) = P(B->H) + P(M -> H) - P(B)*P(M)

#### P(B -> H): necessarily has to be causal power
headache = c(rep(1,18),rep(0,18),rep(1,36))
medicine = c(rep(0,36),rep(1,36))
mod = glm(headache ~ medicine, family = "binomial")
summary(mod)
predict(mod, type="response")

# infection = b0 + b1*vaccine
# b1 =/= vaccine efficacy 
# inverse logit transformation gives change in probability (delta P)
# delta p =/= vaccine efficacy
# CDC: (infection_without_vaccine - infection_with_vaccine)/infection_without_vaccine

# farm: 90% ill to 60% ill (10% healthy to 40% healthy)
animal_id = rep(seq(1,100),2)
location = rep("farm",200)
red_dots = c(rep(1,90),rep(0,10),rep(1,60),rep(0,40))
healthy = c(rep(0,90),rep(1,10),rep(0,60),rep(1,40))
grains = c(rep(0,100),rep(1,100))
leaves = rep(0,200)
treatment = c(rep("None",100),rep("Grain",100))
farm_df = as.data.frame(cbind(animal_id,location,red_dots,healthy,grains,leaves,treatment))

# zoo: 40% ill to 10% ill (60% healthy to 90% healthy)
animal_id = rep(seq(201,300),2)
location = rep("zoo",200)
red_dots = c(rep(1,40),rep(0,60),rep(1,10),rep(0,90))
healthy = c(rep(0,40),rep(1,60),rep(0,10),rep(1,90))
grains = c(rep(0,100),rep(1,100))
leaves = c(rep(0,100),rep(1,100))
treatment = c(rep("None",100),rep("Grains + Leaves",100))
zoo_df = as.data.frame(cbind(animal_id,location,red_dots,healthy,grains,leaves,treatment))

# combined:
combined_df = rbind(farm_df,zoo_df)

# mixed effect logistic regression:
library(lme4)
mod0 = glmer(as.factor(red_dots)~grains+leaves+(1|location/animal_id),data=combined_df,family="binomial")

# logistic regressions:
mod1 = glm(as.numeric(red_dots)~location+leaves+grains,data=combined_df,family="binomial") #combined
mod2 = glm(as.numeric(red_dots)~grains,data=farm_df,family="binomial") # farm only
mod3 = glm(as.numeric(red_dots)~grains,data=zoo_df,family="binomial") # zoo only

#### fully saturated model ####
df = function(sample_size,power1,power2,base_rate){
  n = sample_size/4
  c1 = c(rep(0,n),rep(0,n),rep(1,n),rep(1,n)) # nothing, c1 only, c2 only, both c1 and c2
  c2 = c(rep(0,n),rep(1,n),rep(0,n),rep(1,n))
  px1 = power1 # binary variable
  px2 = power2 # binary variable
  pb = base_rate
  p = 1-(1-px1*c1)*(1-px2*c2)*(1-pb) # union rule
  y = rbinom(sample_size,1,p)        # bernoulli response variable
  return(cbind(c1,c2,y))
}

DF = as.data.frame(df(sample_size=400000,base_rate=6/10,power1=1/3,power2=5/8)) # generate data
names(DF) = c("grain","leaves","healthy")
mean(DF$healthy[DF$grain==0 & DF$leaves==0]) # check base rate ~ 60%
mean(unname(DF$healthy[DF$grain==1 & DF$leaves == 1])) # average healthy when grain + leaves ~ 90%

mod = glm(healthy ~ grain+leaves, family = "binomial", data = DF) # fully saturated model will give the same results
summary(mod) # leaves are more effective

### predicted probabilities
pred_probs = predict.glm(mod,type="response")
control = mean(unname(pred_probs[DF$grain==0 & DF$leaves==0])) # nothing (control)
grain_and_leaves = mean(unname(pred_probs[DF$grain==1 & DF$leaves==1])) # grain + leaves
grain_or_leaves = mean(unname(pred_probs[DF$grain==1 | DF$leaves==1])) # grain or leaves

given_grains1 = mean(unname(pred_probs[DF$grain==1]))
leaves_grains1 = mean(unname(pred_probs[DF$grain==1 & DF$leaves==1]))

### delta P's
deltaP_grain = grain - control # ~ 13%, how should this actually be calculated?
deltaP_leaves = leaves - control # ~ 24, how should this actually be calculated?
deltaP_grain_and_leaves = grain_and_leaves - control # 30% difference between grains + leaves and control

control + deltaP_grain + deltaP_leaves - deltaP_grain*deltaP_leaves - control*deltaP_grain - control*deltaP_leaves + control*deltaP_grain*deltaP_leaves

# Notes:
# maximum predicted probability is 90% (when grain = 1 and leaves = 1)
# minimum predicted probability is 60% (when grain = 0 and leaves = 0)
