#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:08:26 2021

@author: ed203246
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools

###############################################################################
# score_parentedu_byclass

n_stud = 60
class_bias_rnd = [-3, 0, +3]
n_class = len(class_bias_rnd)

edu = np.random.randint(0, 11, n_stud)
b1 = 0.1

classroom_inter = np.concatenate([[val] * int(n_stud / n_class) for val in class_bias_rnd])

#classroom_inter = np.concatenate([np.repeat(np.random.normal(0, 7, 1), int(n_stud / n_class)) for g in range(n_class)])
classroom_lab = np.concatenate([["c%i" % lab] * int(n_stud / n_class) for lab in range(n_class)])

b0 = 10

eps = np.random.normal(0, 1, n_stud)

score = b0 + b1 * edu + classroom_inter + eps

score[score > 20] = 20

df = pd.DataFrame(dict(classroom=classroom_lab, edu=edu, score=score))

agregate = df.groupby('classroom').mean()
print(agregate)
sns.lmplot(x="edu", y="score", data=agregate)

# df.to_csv('datasets/score_parentedu_byclass.csv', index=False)

###############################################################################
# Analysis

df = pd.read_csv('datasets/score_parentedu_byclass.csv')

sns.lmplot(x="edu", y="score", data=df)
lm_glob = smf.ols('score ~ edu', df).fit()
#print(lm_agregate.summary())
print(lm_glob.t_test('edu'))

# Normality of the residuals
lm_diag = pd.DataFrame(dict(prediction=lm_glob.predict(df),
                            residual=lm_glob.resid,
                            classroom=df.classroom))
#sns.scatterplot(x='prediction', y='residual', hue='classroom', data=lm_diag)
sns.displot(lm_diag, x='residual', hue='classroom', kind="kde", fill=True)

###############################################################################
# Agretation

agregate = df.groupby('classroom').mean()
lm_agregate = smf.ols('score ~ edu', agregate).fit()
#print(lm_agregate.summary())
print(lm_agregate.t_test('edu'))

agregate = agregate.reset_index()

fig, axes = plt.subplots(1, 2, figsize=(9, 9))
sns.scatterplot(x='edu', y='score', hue='classroom', data=df, ax=axes[0], s=30)
sns.scatterplot(x='edu', y='score', hue='classroom', data=agregate, ax=axes[0], s=100)
sns.regplot(x="edu", y="score", data=agregate, ax=axes[1])
sns.scatterplot(x='edu', y='score', hue='classroom', data=agregate, ax=axes[1], s=100)


##############################################################################
# Hierarchical Model Specification

# Level 1 model within classes
lv1 = [[classroom_lab, smf.ols('score ~ edu', classroom_df).fit().params['edu']]
 for classroom_lab, classroom_df in df.groupby("classroom")]

lv1 = list()
for classroom_lab, classroom_df in df.groupby("classroom"):
    modlv1 = smf.ols('score ~ edu', classroom_df).fit()
    sse = np.sum(modlv1.resid ** 2)
    lv1.append([classroom_lab, modlv1.params['edu'], sse])

lv1 = pd.DataFrame(lv1, columns=['classroom', 'beta_edu', 'sse'])
print(lv1)

# Level 2 model test beta_edu != 0
lm_hm = smf.ols('beta_edu ~ 1', lv1).fit()
print(lm_hm.t_test('Intercept'))
print("MSE=%.3f" % lm_agregate.mse_resid)

fig, axes = plt.subplots(1, 2, figsize=(9, 9))
for classroom_lab, classroom_df in df.groupby("classroom"):
    sns.regplot(x="edu", y="score", data=classroom_df, ax=axes[0])

g = sns.scatterplot(x=0, y="beta_edu", hue="classroom", data=lv1, ax=axes[1])
axes[1].axhline(0, ls='--')

g = sns.lmplot(x="edu", y="score", hue="classroom", data=df, ax=axes[0])


##############################################################################
# classroom as fix effect

ancova_inter = smf.ols('score ~ edu + classroom', df).fit()
# print(sm.stats.anova_lm(ancova_inter, typ=3))
# print(ancova_inter.summary())
print(ancova_inter.t_test('edu'))
print("MSE=%.3f" % ancova_inter.mse_resid)

# results.loc[len(results)] = ["ANCOVA (biased)"] + list(rmse_coef_tstat_pval(mod=ancova_inter, var='edu'))

# Plot
g = sns.lmplot(x="edu", y="score", hue="classroom", data=df, fit_reg=False)
palette = itertools.cycle(sns.color_palette())
x_jitter = -0.2

# Reression with same slope for all classrooms but a shift, different intercepts
for classroom_lab, classroom_df in df.groupby("classroom"):
    x_ = classroom_df["edu"]
    color = next(palette)
    try:
        group_offset = ancova_inter.params["classroom[T.%s]" % classroom_lab]
    except:
        group_offset = 0
    y_ = ancova_inter.params['Intercept'] + ancova_inter.params['edu'] * x_ + group_offset
    ax = sns.lineplot(x=x_, y=y_, color=color)
    ax.arrow(0+x_jitter, ancova_inter.params['Intercept'], 0, group_offset, head_width=.3, length_includes_head=True, color=color)#, lw=1, fill=False,
    print(classroom_lab, group_offset, color)
    x_jitter += 0.2

mod = ancova_inter

print("Design matrix (independant variables)")
print(mod.model.exog_names)
print(mod.model.exog[:10])

print("Outcome (dependant variable)")
print(mod.model.endog_names)
print(mod.model.endog[:10])

print("Fitted model")
print(mod.params)
sse_ = np.sum(mod.resid ** 2)
df_ = mod.df_resid
mod.df_model
print("MSE %f" % (sse_ / df_), "or", mod.mse_resid)

print("Statistics")
print(mod.tvalues, mod.pvalues)

#results.loc[len(results)] = ["ANCOVA (biased)"] + list(rmse_coef_tstat_pval(mod=ancova_inter, var='edu'))

##############################################################################
# classroom as random effect

lmm_inter = smf.mixedlm("score ~ edu", df, groups=df["classroom"]).fit()
# By defaults use a random intercept for each group.
lmm_inter = smf.mixedlm("score ~ edu", df, groups=df["classroom"], re_formula="~1").fit()

print(lmm_inter.summary())


# Plot
g = sns.lmplot(x="edu", y="score", hue="classroom", data=df, fit_reg=False)
palette = itertools.cycle(sns.color_palette())
x_jitter = -0.2

# Reression with same slope for all classrooms but a shift, different intercepts
for classroom_lab, classroom_df in df.groupby("classroom"):
    x_ = classroom_df["edu"]
    color = next(palette)
    group_offset = lmm_inter.random_effects[classroom_lab][0]
    intercept = lmm_inter.params['Intercept']
    var = lmm_inter.params["Group Var"]
    y_ = lmm_inter.params['Intercept'] + lmm_inter.params['edu'] * x_ + group_offset
    ax = sns.lineplot(x=x_, y=y_, color=color)
    ax.arrow(0+x_jitter, intercept, 0, group_offset, head_width=.3, length_includes_head=True, color=color)#, lw=1, fill=False,
    ax.text(0,  intercept + group_offset, "~N(%.3f, %.2f)" % (intercept, var))

    print(classroom_lab, group_offset, color)
    x_jitter += 0.2

##############################################################################
#
ancova_full = smf.ols('score ~ edu + classroom + edu:classroom', df).fit()
# Full model (including interaction) can use this notation:
# ancova_full = smf.ols('score ~ edu * classroom', df).fit()

# print(sm.stats.anova_lm(lm_fx, typ=3))
# print(lm_fx.summary())
print(ancova_full.t_test('edu'))
print("MSE=%.3f" % ancova_full.mse_resid)

##############################################################################
# classroom as random effect and slope

lmm_full = smf.mixedlm("score ~ edu", df, groups=df["classroom"], re_formula="~1+edu").fit()
print(lmm_full.summary())


