Statistics and Machine Learning in Python
=========================================

- [pdf](ftp://ftp.cea.fr/pub/unati/people/educhesnay/pystatml/StatisticsMachineLearningPython.pdf)
- [www](https://duchesnay.github.io/pystatsml)


Structure
---------

Courses are available in three formats:

1. Jupyter notebooks `*/*.ipynb` files.

2. Python files using [sphinx-gallery](https://sphinx-gallery.github.io/stable/index.html) `*/*.py` files.

3. ReStructuredText files.

All notebooks and python files are converted into `rst` format and then assembled together using sphinx.

Directories and main files:

    introduction/
    ├── machine_learning.rst
    └── python_ecosystem.rst

    python_lang/                        # (Python language)
    ├── python_lang.py # (main file)
    └── python_lang_solutions.py

    scientific_python/
    ├── matplotlib.ipynb
    ├── scipy_numpy.py
    ├── scipy_numpy_solutions.py
    ├── scipy_pandas.py
    └── scipy_pandas_solutions.py

    statistics/                         # (Statistics)
    ├── stat_multiv.ipynb               # (multivariate statistics)
    ├── stat_univ.ipynb                 # (univariate statistics)
    ├── stat_univ_solutions.ipynb
    ├── stat_univ_lab01_brain-volume.py # (lab)
    ├── stat_univ_solutions.ipynb
    └── time_series.ipynb

    machine_learning/                   # (Machine learning)
    ├── clustering.ipynb
    ├── decomposition.ipynb
    ├── decomposition_solutions.ipynb
    ├── linear_classification.ipynb
    ├── linear_regression.ipynb
    ├── manifold.ipynb
    ├── non_linear_prediction.ipynb
    ├── resampling.ipynb
    ├── resampling_solution.py
    └── sklearn.ipynb

    optimization/
    ├── optim_gradient_descent.ipynb
    └── optim_gradient_descent_lab.ipynb

    deep_learning/
    ├── dl_backprop_numpy-pytorch-sklearn.ipynb
    ├── dl_cnn_cifar10_pytorch.ipynb
    ├── dl_mlp_mnist_pytorch.ipynb
    └── dl_transfer-learning_cifar10-ants-


Build
-----

After pulling the repository execute Jupyter notebooks (outputs are expected to be removed before git submission).
```
make exe
```

Build the pdf file (requires LaTeX):
```
make pdf
```

Build the html files:
```
make html
```

Clean everything and  strip output from Jupyter notebook (useless if you installed the nbstripout hook, ):
```
make clean
```

Dependencies
------------
The easier is to install Anaconda at https://www.continuum.io with python >= 3. Anaconda provides

- python 3
- ipython
- Jupyter
- pandoc
- LaTeX to generate pdf

Then install:

1. [sphinx-gallery](https://sphinx-gallery.readthedocs.io)

```
conda install -c conda-forge sphinx-gallery
```

2. [nbstripout](https://github.com/kynan/nbstripout)

```
conda install -c conda-forge nbstripout
```

Configure your git repository with nbstripout pre-commit hook for users who don't want to track output in VCS.

```
nbstripout --install
```

3. Git [LFS](https://git-lfs.github.com/) for datasets

a. Install Git LFS

```
git lfs install
```

b. select the file types you'd like Git LFS to manage

```
git lfs track "*.npz"
git lfs track "*.npy"
git lfs track "*.nii"
git lfs track "*.nii.gz"
git lfs track "*.csv"
```

b. Now make sure .gitattributes is tracked:

```
git add .gitattributes
```

4. LaTeX (optional for pdf)

For Linux debian like:

```
sudo apt-get install latexmk texlive-latex-extra
```

5. MS docx (optional)

[docxbuilder](https://docxbuilder.readthedocs.io/en/latest/docxbuilder.html)

a. Install

```
pip install docxbuilder
pip install docxbuilder[math]
```

b. Build

```
make docx
```
