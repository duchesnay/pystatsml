#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:32:45 2021

@author: edouard.duchesnay@cea.fr
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

def rmse_coef_tstat_pval(mod, var):
    """Return RMSE, Coef, t-tstat and p-value of a fitted model"""
    sse = np.sum(mod.resid ** 2)
    df = mod.df_resid
    return np.sqrt(sse / df), mod.params[var], mod.tvalues[var], mod.pvalues[var]


def plot_lm_diagnosis(residual, prediction, group=None, group_boxplot=False):
    """Regression diagnosis plot"""

    diag_df = pd.DataFrame(dict(prediction=prediction, residual=residual))
    if group is not None:
        diag_df['group'] = group
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
        sns.scatterplot(x='prediction', y='residual', hue='group',data=diag_df,
                        ax=axes[0], legend=False).set_title("Residuals vs pred")
        sns.kdeplot(y='residual', data=diag_df, fill=True,
                    ax=axes[1]).set_title("Residuals")
        if group_boxplot:
            sns.boxplot(y='residual', x='group', data=diag_df,
                ax=axes[2]).set_title("Residuals by group")
        else:
            sns.kdeplot(y='residual', hue='group', data=diag_df, fill=True,
                            ax=axes[2]).set_title("Residuals by group")

    else:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
        sns.scatterplot(x='prediction', y='residual', data=diag_df,
                        ax=axes[0]).set_title("Residuals vs pred")
        sns.kdeplot(y='residual', data=diag_df, fill=True,
                    ax=axes[1]).set_title("Residuals")


def plot_ancova_oneslope_grpintercept(x, y, group, df, model, palette=sns.color_palette()):
    """Plot Ancova model:  single slope with group intercepts

    Parameters
    ----------
    x : str
        Independant covariable.
    y : str
        Dependant variable.
    group : str
        Independant factor.
    df : DataFrame
        data.
    model : statsmodels.regression.linear_model
        Fitted model.
    palette : palette, optional
        The default is sns.color_palette().

    Returns
    -------
    None.
    """
    _ = sns.lmplot(x=x, y=y, hue=group, data=df, fit_reg=False)
    palette = itertools.cycle(palette)
    x_jitter = -0.2

    # Reression with same slope for all groups but a shift, different intercepts
    for group_lab, group_df in df.groupby(group):
        # print(group_lab)
        x_ = group_df[x]
        color = next(palette)
        try:
            group_offset = model.params["%s[T.%s]" % (group, group_lab)]
        except:
            group_offset = 0
        y_pred= model.params['Intercept'] + model.params[x] * x_ + group_offset
        ax = sns.lineplot(x=x_, y=y_pred, color='k')
        ax.arrow(0+x_jitter, model.params['Intercept'], 0, group_offset, head_width=.3,
                 length_includes_head=True, color=color)
        x_jitter += 0.2


def plot_lmm_oneslope_randintercept(x, y, group, df, model,
                                    palette=sns.color_palette(), add_text=True):
    """Plot LMM: single slope with random intercepts

    Parameters
    ----------
    x : str
        Independant covariable.
    y : str
        Dependant variable.
    group : str
        Independant factor.
    df : DataFrame
        data.
    model : statsmodels.regression.linear_model
        Fitted model.
    palette : palette, optional
        The default is sns.color_palette().

    Returns
    -------
    None.
    """
    _ = sns.lmplot(x=x, y=y, hue=group, data=df, fit_reg=False)
    palette = itertools.cycle(palette)
    x_jitter = -0.2
    intercept = model.params['Intercept']
    var = model.params["Group Var"]

    # Reression with same slope for all group but with different intercepts
    for group_lab, group_df in df.groupby(group):
        x_ = group_df[x]
        color = next(palette)
        group_offset = model.random_effects[group_lab][0]
        y_ = model.params['Intercept'] + model.params[x] * \
            x_ + group_offset
        ax = sns.lineplot(x=x_, y=y_, color=color, legend=False)
        ax.arrow(0 + x_jitter, intercept, 0, group_offset, head_width=.3,
                 length_includes_head=True, color=color)
        if add_text:
            ax.text(0,  intercept + group_offset, "~N(%.3f, %.2f)" % (intercept, var))
        x_jitter += 0.2


def plot_ancova_fullmodel(x, y, group, df, model, palette=sns.color_palette()):
    """Plot Ancova full model: y ~ x + group + x:group

    Parameters
    ----------
    x : str
        Independant covariable.
    y : str
        Dependant variable.
    group : str
        Independant factor.
    df : DataFrame
        data.
    model : statsmodels.regression.linear_model
        Fitted model.
    palette : palette, optional
        The default is sns.color_palette().

    Returns
    -------
    None.
    """

    _ = sns.lmplot(x=x, y=y, hue=group, data=df, fit_reg=False)
    palette = itertools.cycle(palette)
    x_jitter = -0.2

    for group_lab, group_df in df.groupby(group):
        x_ = group_df[x]
        color = next(palette)
        try:
            group_offset = model.params["%s[T.%s]" % (group, group_lab)]
        except:
            group_offset = 0
        y_ = model.params['Intercept'] + model.params['edu'] * \
            x_ + group_offset
        ax = sns.lineplot(x=x_, y=y_, color='k', linestyle='--')
        y_ = model.predict(df)
        ax = sns.lineplot(x=x_, y=y_, color=color)
        ax.arrow(0+x_jitter, model.params['Intercept'], 0, group_offset,
                 head_width=.3, length_includes_head=True, color=color)
        x_jitter += 0.2