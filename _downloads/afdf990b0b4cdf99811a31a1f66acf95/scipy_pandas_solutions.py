"""
Exercises: Pandas: data manipulation
------------------------------------

Data Frame
~~~~~~~~~~

1. Read the iris dataset at 'https://github.com/neurospin/pystatsml/tree/master/datasets/iris.csv'

2. Print column names

3. Get numerical columns

4. For each species compute the mean of numerical columns and store it in  a ``stats`` table like:

::

          species  sepal_length  sepal_width  petal_length  petal_width
    0      setosa         5.006        3.428         1.462        0.246
    1  versicolor         5.936        2.770         4.260        1.326
    2   virginica         6.588        2.974         5.552        2.026


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


url = 'https://github.com/duchesnay/pystatsml/raw/master/datasets/iris.csv'
df = pd.read_csv(url)

num_cols = df._get_numeric_data().columns

stats = list()

for grp, d in df.groupby("species"):
    print(grp)
    #print()
    stats.append( [grp] + d.loc[:, num_cols].mean(axis=0).tolist())

stats = pd.DataFrame(stats, columns=["species"] + num_cols.tolist())
print(stats)

# or
df.groupby("species").mean()

##

df.loc[[0, 1] ,"petal_width"] = None

df.petal_width

df["petal_width"][df["petal_width"].isnull()] = \
    df["petal_width"][df["petal_width"].notnull()].median()


#

l = [(1, "a", 1), (2, "b", 2)]

for x, y, z in l:
    print(x, y, z)




















