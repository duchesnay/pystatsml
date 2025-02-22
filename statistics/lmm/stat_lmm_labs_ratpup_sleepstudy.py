'''
Hands-On: Linear Mixed Models
=============================

Two labs on Linear Mixed Models:

-
-

Statsmodels links:

- [MixedLM](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html#statsmodels.regression.mixed_linear_model.MixedLM)
- [formula](https://www.statsmodels.org/stable/generated/statsmodels.formula.api.mixedlm.html#statsmodels.formula.api.mixedlm)
- [Examples](https://www.statsmodels.org/stable/mixed_linear.html)

'''

import os
os.chdir('/home/ed203246/git/rstats/stat21_lmm')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from stat_lmm_utils import rmse_coef_tstat_pval
from stat_lmm_utils import plot_lm_diagnosis
from stat_lmm_utils import plot_ancova_oneslope_grpintercept
from stat_lmm_utils import plot_lmm_oneslope_randintercept
from stat_lmm_utils import plot_ancova_fullmodel


###############################################################################
# Ratpup dataset: Two-Level Models for Clustered Data
# ---------------------------------------------------
#
# Ratpup dataset: https://rdrr.io/rforge/WWGbook/man/ratpup.html
#
# Quote from [Brady et al. 2014]: The Rat Pup data is an example of a two-level
# clustered data set obtained from a cluster-
# randomized trial: each litter (cluster) was randomly assigned to a specific level of treatment,
# and rat pups (units of analysis) were nested within litters. The birth weights of rat
# pups within the same litter are likely to be correlated because the pups shared the same
# maternal environment. In models for the Rat Pup data, we include random litter effects
# (which imply that observations on the same litter are correlated) and fixed effects asso-
# ciated with treatment. Our analysis uses a two-level random intercept model to compare
# the mean birth weights of rat pups from litters assigned to the three different doses, after
# taking into account variation both between litters and between pups within the same
# litter. Ratpup is a two-level (rat level 1, and litter level 2) random effect model.
#
# Variables:
#
# - pup.id: Unique identifier for each rat pup
# - weight: Birth weight of the rat pup (the dependent variable)
# - sex: Sex of the rat pup (Male, Female)
# - litter: Litter ID number
# - litsize: Litter size (i.e., number of pups per litter)
# - treatment: Dose level of the experimental compound assigned to the litter (High, Low, Control)
#
# **Objective: Explore Sex effect on weight**

ratpup = pd.read_csv("datasets/rat_pup.csv")
ratpup["sex1"] = 0
ratpup.loc[ratpup.sex == "Female", "sex1"] = 1
ratpup["litter"] = ratpup["litter"].astype("category")
sns.histplot(ratpup.weight)
results = pd.DataFrame(columns=["Model", "RMSE", "Coef", "Stat", "Pval"])

###############################################################################
# Global sex effect biased
# ~~~~~~~~~~~~~~~~~~~~~~~~

sns.boxplot(x="sex", y="weight", data = ratpup)
lm_glob = smf.ols('weight ~ sex', ratpup).fit()
print(sm.stats.anova_lm(lm_glob, typ=2))
# or
#test_glob = lm_glob.t_test('sex[T.Male]')
results.loc[len(results)] = ["LM GLOB (biased)"] + list(rmse_coef_tstat_pval(mod=lm_glob, var='sex[T.Male]'))

###############################################################################
#
# Residuals diagnosis

plot_lm_diagnosis(residual=lm_glob.resid, prediction=lm_glob.predict(ratpup), group=ratpup.litter, group_boxplot=True)
sns.boxplot(x=ratpup.litter, y=lm_glob.resid)


###############################################################################
# Model a litter intercept as a fixed effect: ANCOVA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sns.boxplot(x = "litter", y = "weight", hue="sex", data = ratpup)
ancova_inter = smf.ols('weight ~ sex + litter', ratpup).fit()

###############################################################################
# Model internals

print("## Design matrix (independant variables):")
print(ancova_inter.model.exog_names)
print(ancova_inter.model.exog[:15, :20])

# print(sm.stats.anova_lm(ancova_inter, typ=3))
# print(ancova_inter.summary())
print(ancova_inter.t_test('sex[T.Male]'))

print("MSE=%.3f" % ancova_inter.mse_resid)
results.loc[len(results)] = ["ANCOVA-Inter (biased)"] + list(rmse_coef_tstat_pval(mod=ancova_inter, var='sex[T.Male]'))


###############################################################################
# Residuals diagnosis

plot_lm_diagnosis(residual=ancova_inter.resid, prediction=ancova_inter.predict(ratpup), group=ratpup.litter, group_boxplot=True)
_ = sns.boxplot(x=ratpup.litter, y=ancova_inter.resid)


###############################################################################
# Hierarchical model
# ~~~~~~~~~~~~~~~~~~

# sns.boxplot(x = "litter", y = "weight", hue="sex", data = ratpup)
# or
# grid = sns.catplot(x="sex", y="weight", col="litter", data=ratpup, col_wrap=4)#, sharex=False, col_wrap=4, data=ratpup, height=4)
lv1 = list()

for lit, df in ratpup.groupby("litter"):
    # Level 1 Model (Rat Pup)
    lm_lv1 = smf.ols('weight ~ sex', df).fit()
    try:
        lv1.append(lm_lv1.params['sex[T.Male]'])
    except:
        print(lit)
        pass

lv1 = pd.DataFrame(lv1, columns=['Male'])
lm_hm = smf.ols('Male ~ 1', lv1).fit()
print(lm_hm.summary())

# test_lm_hm = lm_hm.t_test('Intercept')
results.loc[len(results)] = ["LM Hierarchical"] + list(rmse_coef_tstat_pval(mod=lm_hm, var='Intercept'))


###############################################################################
# Model the litter random intercept: linear mixed model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

lmm_inter = smf.mixedlm("weight ~ sex", ratpup, groups=ratpup["litter"]).fit()
print(lmm_inter.summary())


results.loc[len(results)] = ["LMM-Inter"] + list(rmse_coef_tstat_pval(mod=lmm_inter, var='sex[T.Male]'))

###############################################################################
# Explore model

print("Fixed effect:")
print(lmm_inter.params)

print("Random effect:")
print(lmm_inter.random_effects)


###############################################################################
# Results
# ~~~~~~~

print(results)


###############################################################################
# Sleepstudy dataset: Longitudinal study
# --------------------------------------
#
# The average reaction time per day for subjects in a sleep deprivation study.
# On day 0 the subjects had their normal amount of sleep.
# Starting that night they were restricted to 3 hours of sleep per night.
# The observations represent the average reaction time on a series of tests given each day to each subject.
# A data frame with 180 observations on the following 3 variables.
#
# - Reaction: Average reaction time (ms)
# - Days: Number of days of sleep deprivation
# - Subject: Subject number on which the observation was made.
#
# This is a two level model:
#
# - Level 1 (within subject at time level): _Days_ (fixed effect independant var) and _Reaction_ (dependant var.)
# - Level 2 (between subjects at subject level): _Subject_ (mixed effect)
#
# **Objective: model the effects of time (_Days_) on _Reaction_ time.**


sleepstudy = pd.read_csv("datasets/sleepstudy.csv")
sleepstudy["Subject"] = sleepstudy["Subject"].astype("category")
sns.histplot(sleepstudy.Reaction)
results = pd.DataFrame(columns=["Model", "RMSE", "Coef", "Stat", "Pval"])

###############################################################################
#
# Global Days effect biased

sns.lmplot(x="Days", y="Reaction", data=sleepstudy)
lm_glob = smf.ols('Reaction ~ Days', sleepstudy).fit()
print(sm.stats.anova_lm(lm_glob, typ=2))
# or
#test_glob = lm_glob.t_test('sex[T.Male]')

results.loc[len(results)] = ["LM GLOB"] + list(rmse_coef_tstat_pval(mod=lm_glob, var='Days'))

###############################################################################
# Model a Subject intercept as a fixed effect: ANCOVA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ancova_inter = smf.ols('Reaction ~ Days + Subject', sleepstudy).fit()
print(sm.stats.anova_lm(ancova_inter, typ=2))
# or
# test_fx = lm_fx.t_test('sex[T.Male]')

results.loc[len(results)] = ["ANCOVA-Inter (biased)"] + list(rmse_coef_tstat_pval(mod=ancova_inter, var='Days'))


###############################################################################
# Plot limited to 3 groups

sleepstudy_ = sleepstudy[sleepstudy['Subject'].isin(sleepstudy['Subject'].unique()[:3])]
sleepstudy_.loc[:, 'Subject'] = sleepstudy_['Subject'].cat.remove_unused_categories()
plot_ancova_oneslope_grpintercept(x="Days", y="Reaction", group="Subject",
                                  df=sleepstudy_, model=ancova_inter)

###############################################################################
#  Hierarchical/multilevel modeling
# ---------------------------------

x, y, group, df = 'Days', 'Reaction', 'Subject', sleepstudy

# Level 1 model within subject
lv1 = [[group_lab, smf.ols('%s ~ %s' % (y, x), group_df).fit().params[x]]
       for group_lab, group_df in df.groupby(group)]

lv1 = pd.DataFrame(lv1, columns=[group, 'beta'])
print(lv1)

# Level 2 model test beta_Days != 0
lm_hm = smf.ols('beta ~ 1', lv1).fit()
print(lm_hm.t_test('Intercept'))
print("MSE=%.3f" % lm_hm.mse_resid)

results.loc[len(results)] = ["Hierarchical"] + \
    list(rmse_coef_tstat_pval(mod=lm_hm, var='Intercept'))


fig, axes = plt.subplots(1, 2, figsize=(9, 6))
for group_lab, group_df in df.groupby(group):
    sns.regplot(x=x, y=y, data=group_df, ax=axes[0])

axes[0].set_title("Level 1: Regressions within %s" % group)
ax = sns.barplot(x=group, y="beta", hue=group, data=lv1, ax=axes[1])
axes[1].axhline(0, ls='--')
axes[1].text(0, 0, "Null slope")
_ = axes[1].set_title("Level 2: Test Slopes between classrooms")


###############################################################################
# Model the subject random intercept: linear mixed model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

lmm_inter = smf.mixedlm("Reaction ~ Days", sleepstudy, groups=sleepstudy["Subject"]).fit()
print(lmm_inter.summary())


results.loc[len(results)] = ["LMM-Inter"] + \
    list(rmse_coef_tstat_pval(mod=lmm_inter, var='Days'))


print("Fixed effect:")
print(lmm_inter.params)

print("Random effect:")
print(lmm_inter.random_effects)

intercept = lmm_inter.params['Intercept']
var = lmm_inter.params["Group Var"]


###############################################################################
# Model the subject intercept and slope as a fixed effect: ANCOVA with interactions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ancova_full = smf.ols('Reaction ~ Days + Subject + Days:Subject', sleepstudy).fit()


print(ancova_full.t_test('Days'))
print("MSE=%.3f" % ancova_full.mse_resid)
results.loc[len(results)] = ["ANCOVA-Full (biased)"] + \
    list(rmse_coef_tstat_pval(mod=ancova_full, var='Days'))

###############################################################################
# Model the subject random intercept and slope with LMM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

lmm_full = smf.mixedlm("Reaction ~ Days", sleepstudy, groups=sleepstudy["Subject"], re_formula="~Days").fit()
print(lmm_full.summary())
# fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)

results.loc[len(results)] = ["LMM-Full (biased)"] + \
    list(rmse_coef_tstat_pval(mod=lmm_full, var='Days'))


###############################################################################
# Results

print(results)

###############################################################################
# References
#
# - Brady et al. 2014: Brady T. West, Kathleen B. Welch, Andrzej T. Galecki,
# [Linear Mixed Models: A Practical Guide Using Statistical Software (2nd Edition)]
# (http://www-personal.umich.edu/~bwest/almmussp.html), 2014