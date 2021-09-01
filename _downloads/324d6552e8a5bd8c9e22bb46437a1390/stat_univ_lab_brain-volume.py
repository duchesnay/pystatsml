'''
Lab: Brain volumes study
=======================

The study provides the brain volumes of grey matter (gm), white matter (wm)
and cerebrospinal fluid) (csf) of 808 anatomical MRI scans.
'''

###############################################################################
# Manipulate data
# ---------------

###############################################################################
# Set the working directory within a directory called "brainvol"
#
# Create 2 subdirectories: `data` that will contain downloaded data and
# `reports` for results of the analysis.

import os
import os.path
import pandas as pd
import tempfile
import urllib.request

WD = os.path.join(tempfile.gettempdir(), "brainvol")
os.makedirs(WD, exist_ok=True)
#os.chdir(WD)

# use cookiecutter file organization
# https://drivendata.github.io/cookiecutter-data-science/
os.makedirs(os.path.join(WD, "data"), exist_ok=True)
#os.makedirs("reports", exist_ok=True)

###############################################################################
# **Fetch data**
#
# - Demographic data `demo.csv` (columns: `participant_id`, `site`, `group`,
#   `age`, `sex`) and tissue volume data: `group` is Control or Patient.
#   `site` is the recruiting site.
# - Gray matter volume `gm.csv` (columns: `participant_id`, `session`, `gm_vol`)
# - White matter volume `wm.csv` (columns: `participant_id`, `session`, `wm_vol`)
# - Cerebrospinal Fluid `csf.csv` (columns: `participant_id`, `session`, `csf_vol`)

base_url = 'https://github.com/duchesnay/pystatsml/raw/master/datasets/brain_volumes/%s'
data = dict()
for file in ["demo.csv", "gm.csv", "wm.csv", "csf.csv"]:
    urllib.request.urlretrieve(base_url % file, os.path.join(WD, "data", file))

# Read all CSV in one line
# dicts = {k: pd.read_csv(os.path.join(WD, "data", "%s.csv" % k))
#          for k in ["demo", "gm", "wm", "csf"]}

demo = pd.read_csv(os.path.join(WD, "data", "demo.csv"))
gm = pd.read_csv(os.path.join(WD, "data", "gm.csv"))
wm = pd.read_csv(os.path.join(WD, "data", "wm.csv"))
csf = pd.read_csv(os.path.join(WD, "data", "csf.csv"))

print("tables can be merge using shared columns")
print(gm.head())

###############################################################################
# **Merge tables** according to `participant_id`

brain_vol = pd.merge(pd.merge(pd.merge(demo, gm), wm), csf)
assert brain_vol.shape == (808, 9)

###############################################################################
# **Drop rows with missing values**

brain_vol = brain_vol.dropna()
assert brain_vol.shape == (766, 9)

###############################################################################
# **Compute Total Intra-cranial volume**
# `tiv_vol` = `gm_vol` + `csf_vol` + `wm_vol`.

brain_vol["tiv_vol"] = brain_vol["gm_vol"] + brain_vol["wm_vol"] + brain_vol["csf_vol"]

###############################################################################
# **Compute tissue fractions**
# `gm_f = gm_vol / tiv_vol`, `wm_f  = wm_vol / tiv_vol`.

brain_vol["gm_f"] = brain_vol["gm_vol"] / brain_vol["tiv_vol"]
brain_vol["wm_f"] = brain_vol["wm_vol"] / brain_vol["tiv_vol"]

###############################################################################
# **Save in a excel file** `brain_vol.xlsx`

brain_vol.to_excel(os.path.join(WD, "data", "brain_vol.xlsx"),
                   sheet_name='data', index=False)

###############################################################################
# Descriptive Statistics
# ----------------------

###############################################################################
# Load excel file `brain_vol.xlsx`

import os
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smfrmla
import statsmodels.api as sm

brain_vol = pd.read_excel(os.path.join(WD, "data", "brain_vol.xlsx"),
                          sheet_name='data')
# Round float at 2 decimals when printing
pd.options.display.float_format = '{:,.2f}'.format


###############################################################################
# **Descriptive statistics**
# Most of participants have several MRI sessions (column `session`)
# Select on rows from session one "ses-01"

brain_vol1 = brain_vol[brain_vol.session == "ses-01"]
# Check that there are no duplicates
assert len(brain_vol1.participant_id.unique()) == len(brain_vol1.participant_id)


###############################################################################
# Global descriptives statistics of numerical variables

desc_glob_num = brain_vol1.describe()
print(desc_glob_num)


###############################################################################
# Global Descriptive statistics of categorical variable

desc_glob_cat = brain_vol1[["site", "group", "sex"]].describe(include='all')
print(desc_glob_cat)

print("Get count by level")
desc_glob_cat = pd.DataFrame({col: brain_vol1[col].value_counts().to_dict()
                             for col in ["site", "group", "sex"]})
print(desc_glob_cat)


###############################################################################
# Remove the single participant from site 6

brain_vol = brain_vol[brain_vol.site != "S6"]
brain_vol1 = brain_vol[brain_vol.session == "ses-01"]
desc_glob_cat = pd.DataFrame({col: brain_vol1[col].value_counts().to_dict()
                             for col in ["site", "group", "sex"]})
print(desc_glob_cat)


###############################################################################
# Descriptives statistics of numerical variables per clinical status
desc_group_num = brain_vol1[["group", 'gm_vol']].groupby("group").describe()
print(desc_group_num)


###############################################################################
# Statistics
# ----------
#
# Objectives:
#
# 1. Site effect of gray matter atrophy
# 2. Test the association between the age and gray matter atrophy in the control
#    and patient population independently.
# 3. Test for differences of atrophy between the patients and the controls
# 4. Test for interaction between age and clinical status, ie: is the brain
#    atrophy process in patient population faster than in the control population.
# 5. The effect of the medication in the patient population.

import statsmodels.api as sm
import statsmodels.formula.api as smfrmla
import scipy.stats
import seaborn as sns

###############################################################################
# **1 Site effect on Grey Matter atrophy**
#
# The model  is Oneway Anova gm_f ~ site
# The ANOVA test has important assumptions that must be satisfied in order
# for the associated p-value to be valid.
#
# - The samples are independent.
# - Each sample is from a normally distributed population.
# - The population standard deviations of the groups are all equal.
#   This property is known as homoscedasticity.
#

###############################################################################
# Plot
sns.violinplot(x="site", y="gm_f", data=brain_vol1)
# sns.violinplot(x="site", y="wm_f", data=brain_vol1)

###############################################################################
# Stats with scipy

fstat, pval = scipy.stats.f_oneway(*[brain_vol1.gm_f[brain_vol1.site == s]
                                   for s in brain_vol1.site.unique()])
print("Oneway Anova gm_f ~ site F=%.2f, p-value=%E" % (fstat, pval))

###############################################################################
# Stats with statsmodels

anova = smfrmla.ols("gm_f ~ site", data=brain_vol1).fit()
# print(anova.summary())
print("Site explains %.2f%% of the grey matter fraction variance" %
      (anova.rsquared * 100))

print(sm.stats.anova_lm(anova, typ=2))

###############################################################################
# **2. Test the association between the age and gray matter atrophy** in the
# control and patient population independently.

###############################################################################
# Plot

sns.lmplot(x="age", y="gm_f", hue="group", data=brain_vol1)

brain_vol1_ctl = brain_vol1[brain_vol1.group == "Control"]
brain_vol1_pat = brain_vol1[brain_vol1.group == "Patient"]

###############################################################################
# Stats with scipy

print("--- In control population ---")
beta, beta0, r_value, p_value, std_err = \
    scipy.stats.linregress(x=brain_vol1_ctl.age, y=brain_vol1_ctl.gm_f)

print("gm_f = %f * age + %f" % (beta, beta0))
print("Corr: %f, r-squared: %f, p-value: %f, std_err: %f"\
      % (r_value, r_value**2, p_value, std_err))

print("--- In patient population ---")
beta, beta0, r_value, p_value, std_err = \
    scipy.stats.linregress(x=brain_vol1_pat.age, y=brain_vol1_pat.gm_f)

print("gm_f = %f * age + %f" % (beta, beta0))
print("Corr: %f, r-squared: %f, p-value: %f, std_err: %f"\
      % (r_value, r_value**2, p_value, std_err))

print("Decrease seems faster in patient than in control population")

###############################################################################
# Stats with statsmodels

print("--- In control population ---")
lr = smfrmla.ols("gm_f ~ age", data=brain_vol1_ctl).fit()
print(lr.summary())
print("Age explains %.2f%% of the grey matter fraction variance" %
      (lr.rsquared * 100))

print("--- In patient population ---")
lr = smfrmla.ols("gm_f ~ age", data=brain_vol1_pat).fit()
print(lr.summary())
print("Age explains %.2f%% of the grey matter fraction variance" %
      (lr.rsquared * 100))

###############################################################################
# Before testing for differences of atrophy between the patients ans the controls
# **Preliminary tests for age x group effect** (patients would be older or
# younger than Controls)

###############################################################################
# Plot

sns.violinplot(x="group", y="age", data=brain_vol1)

###############################################################################
# Stats with scipy

print(scipy.stats.ttest_ind(brain_vol1_ctl.age, brain_vol1_pat.age))

###############################################################################
# Stats with statsmodels

print(smfrmla.ols("age ~ group", data=brain_vol1).fit().summary())
print("No significant difference in age between patients and controls")

###############################################################################
# **Preliminary tests for sex x group** (more/less males in patients than
# in Controls)

crosstab = pd.crosstab(brain_vol1.sex, brain_vol1.group)
print("Obeserved contingency table")
print(crosstab)

chi2, pval, dof, expected = scipy.stats.chi2_contingency(crosstab)

print("Chi2 = %f, pval = %f" % (chi2, pval))
print("No significant difference in sex between patients and controls")

###############################################################################
# **3. Test for differences of atrophy between the patients and the controls**

print(sm.stats.anova_lm(smfrmla.ols("gm_f ~ group", data=brain_vol1).fit(),
                        typ=2))
print("No significant difference in atrophy between patients and controls")

###############################################################################
# This model is simplistic we should adjust for age and site

print(sm.stats.anova_lm(smfrmla.ols(
        "gm_f ~ group + age + site", data=brain_vol1).fit(), typ=2))
print("No significant difference in GM between patients and controls")

###############################################################################
# Observe age effect

###############################################################################
# **4. Test for interaction between age and clinical status**, ie: is the brain
# atrophy process in patient population faster than in the control population.

ancova = smfrmla.ols("gm_f ~ group:age + age + site", data=brain_vol1).fit()
print(sm.stats.anova_lm(ancova, typ=2))

print("= Parameters =")
print(ancova.params)

print("%.3f%% of grey matter loss per year (almost %.1f%% per decade)" %
      (ancova.params.age * 100, ancova.params.age * 100 * 10))

print("grey matter loss in patients is accelerated by %.3f%% per decade" %
      (ancova.params['group[T.Patient]:age'] * 100 * 10))
