Statistics and Machine Learning in Python
=========================================

- [pdf](https://raw.github.com/duchesnay/data/master/pdf/StatisticsMachineLearningPython.pdf)
- [www](https://duchesnay.github.io/pystatsml)


Structure
---------

The Course is a [Sphinx project](https://www.sphinx-doc.org/en/master) made of :

1. Jupyter notebooks `*/*.ipynb` files.

2. Python files using [sphinx-gallery](https://sphinx-gallery.github.io/stable/index.html) `*/*.py` files.

3. ReStructuredText or Markdown files.

All notebooks and python files are converted into `rst` format and then assembled together using sphinx.

Directories and main files:
    introduction/
    └── introduction_python_for_datascience.rst

    python_lang/
    ├── python_lang.py
    └── python_lang_solutions.py

    data_manipulation/
    ├── data_numpy.py
    ├── data_numpy_solutions.py
    ├── data_pandas.py
    ├── data_pandas_solutions.py
    └── data_visualization.ipynb

    numerical_methods/
    ├── optim_gradient_descent.ipynb
    ├── numerical_differentiation_integration.ipynb
    ├── symbolic_maths.ipynb
    ├── symbolic_maths.rst
    ├── time_series.ipynb
    └── time_series.rst

    statistics/
    ├── stat_univ.ipynb
    ├── stat_univ_solutions.ipynb
    ├── stat_multiv.ipynb
    ├── stat_multiv_solutions.py
    ├── lmm
    │   └── lmm.ipynb
    └── stat_montecarlo.ipynb

    ml_unsupervised/
    ├── clustering.ipynb
    ├── introduction_to_ml.rst
    ├── linear_dimensionality_reduction.ipynb
    ├── linear_dimensionality_reduction_solutions.ipynb
    ├── manifold_learning.ipynb
    └── manifold_learning_solutions.ipynb

    ml_supervised/
    ├── overfitting.ipynb
    ├── ensemble_learning.py
    ├── kernel_svm.py
    ├── linear_classification.ipynb
    └── linear_regression.ipynb

    deep_learning/
    ├── dl_backprop_numpy-pytorch-sklearn.ipynb
    ├── dl_mlp_pytorch.ipynb
    ├── dl_cnn_cifar10_pytorch.ipynb
    └── dl_cnn-pretraining_pytorch.rst


Installation for students
-------------------------

Clone the repository
~~~~~~~~~~~~~~~~~~~~

```
git clone https://github.com/duchesnay/pystatsml.git
cd pystatsml
```

Using Anaconda
~~~~~~~~~~~~~~

Install [Anaconda](https://www.anaconda.com) with python >= 3.

Standard user (student) should install the required data analysis packages.
Create and activate the `pystatsml_student` environment:

```
conda env create -f environment_student.yml
conda activate pystatsml_student
```

Usinf Pixi
~~~~~~~~~~

Install [Pixi](https://pixi.sh/latest/)

Linux & macOS

```
curl -fsSL https://pixi.sh/install.sh | bash
```

Windows

```
iwr -useb https://pixi.sh/install.ps1 | iex
```


Install dependencies contained in pixi.toml file (within the project directory)

```
pixi install
```

Activate an environment (within the project directory)

```
pixi shell
```

What’s in the environment?

```
pixi list
```

Deactivating an environment

```
exit
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

Contributing
------------

Cross-references
~~~~~~~~~~~~~~~~

Add the `ref:` prefix to your references.

Defining references label:

- Markdown file (add `#` before the label)

```
### Demonstration of Negative Log-Likelihood (NLL) {#ref:demonstration-nll}
```

Cross-referencing:

- Jupyter Notebook

```
[Demonstration of Negative Log-Likelihood (NLL)](ref:demonstration-nll)
```