Statistics and Machine Learning in Python
=========================================

- [pdf](https://raw.github.com/duchesnay/data/master/pdf/StatisticsMachineLearningPython.pdf)
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


Installation for students
-------------------------

Install Anaconda at https://www.anaconda.com/ with python >= 3.

Standard user (student) should install the required data analysis packages.
Create and activate the `pystatsml_student` environment:

```
conda env create -f environment_student.yml
conda activate pystatsml_student
```

Installation for teachers: to build the documents
-------------------------------------------------

Expert users (teachers) who need to build (pdf, html, etc.) the course should install additional packages including:

- pandoc
- [sphinx-gallery](https://sphinx-gallery.readthedocs.io)
- [nbstripout](https://github.com/kynan/nbstripout)

Create and activate the ``pystatsml_teacher`` environment:

```
conda env create -f environment_teacher.yml
conda activate pystatsml_teacher
```

Build the documents.
Configure your git repository with `nbstripout`: a pre-commit hook for users who don't want to track notebooks' outputs in git.

```
nbstripout --install
```

Optional: install LaTeX to generate pdf. For Linux debian like:

```
sudo apt-get install latexmk texlive-latex-extra
```


After pulling the repository execute Jupyter notebooks (outputs are expected to be removed before git submission):

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

Clean everything:

```
make clean
```

Optional to generate  Microsoft docx. Use [docxbuilder](https://docxbuilder.readthedocs.io/en/latest/docxbuilder.html):

```
make docx
```

