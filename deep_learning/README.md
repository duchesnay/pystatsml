# Course

## Introduction to Deep Learning

- [Slides:](https://m2dsupsdlclass.github.io/lectures-labs/slides/01_intro_to_deep_learning/index.html)

## 1. Optimization: Gradient descent and Backpropagation

- [Slides:](https://m2dsupsdlclass.github.io/lectures-labs/slides/02_backprop/index.html)

- [Lab: `dl_optim-backprop_numpy-pytorch-sklearn.ipynb`](https://github.com/duchesnay/pystatsml/tree/master/deep_learning/dl_optim-backprop_numpy-pytorch-sklearn.ipynb)

## 2. Multi-Layer Perceptron

- [Lab: `dl_mlp_mnist_pytorch.ipynb`](https://github.com/duchesnay/pystatsml/tree/master/deep_learning/dl_mlp_mnist_pytorch.ipynb)


## 3. Convolutional Neural Networks (CNN)

### Architectures

- [CNN architeures](https://www.jeremyjordan.me/convnet-architectures/)

- [Slides:](https://m2dsupsdlclass.github.io/lectures-labs/slides/04_conv_nets/index.html)

-  [Lab: `dl_cnn_mnist_pytorch.ipynb`](https://github.com/duchesnay/pystatsml/tree/master/deep_learning/dl_cnn_mnist_pytorch.ipynb)


## 4. Transfer Learning

-  [Lab: `dl_transfer-learning_ants-bees_pytorch.ipynb`](https://github.com/duchesnay/pystatsml/tree/master/deep_learning/dl_transfer-learning_ants-bees_pytorch.ipynb)

## 5. Deep Learning for Object Detection and Image Segmentation


# Ressources

## Deep Learning class, Master Datascience Paris Saclay

[Deep Learning class, Master Datascience Paris Saclay](https://github.com/m2dsupsdlclass/lectures-labs)

## Stanford ML courses

- [Deep learning - cs-231n @stanford.edu](http://cs231n.stanford.edu/)

- [Deep Learning Cheatsheet - cs-230 @stanford.edu](https://stanford.edu/~shervine/teaching/cs-230/)

- [Machine Learning Cheatsheet - cs-229 @stanford.edu](https://stanford.edu/~shervine/teaching/cs-229/)


## Anaconda

Download from  [www.anaconda.com](https://www.anaconda.com/)

Choose Python 3.x

Update conda

    conda update -n base -c defaults conda

## Pytorch


- [WWW tutorials](https://pytorch.org/tutorials/)

- [github tutorials](https://github.com/pytorch/tutorials)

- [github examples](https://github.com/pytorch/examples)


### Installation

[pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)


**Anaconda + No CUDA**

    conda install pytorch-cpu torchvision-cpu -c pytorch

Check if torch can be loaded. If CUDA is not available, we will use CPU instead of GPU.

    python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

**Anaconda + CUDA 10:**

    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch


## Optional install Keras for Deep Learning class, Master Datascience Paris Saclay

[Deep Learning class](https://github.com/m2dsupsdlclass/lectures-labs)

Create an new environement called ``py36`` where we will install python 3.6 for Keras and tensor flow

    conda create --name py36
    conda activate py36


[installation instructions](https://github.com/m2dsupsdlclass/lectures-labs/blob/master/installation_instructions.md)

Open a console / terminal and update the following packages with conda:

    conda activate py36
    conda install python=3.6 numpy scikit-learn jupyter ipykernel matplotlib pip
    conda install pandas h5py pillow scikit-image lxml tensorflow keras

Check that you can import tensorflow with the python from anaconda:

    python3 -c "import tensorflow as tf; print(tf.__version__)"

If you have several installations of Python on your system (virtualenv, conda environments...), it can be confusing to select the correct Python environment from the jupyter interface. You can name this environment for instance "py36" and reference it as a Jupyter kernel:

    python3 -m ipykernel install --user --name py36 --display-name py36

To take pictures with the webcam we will also need opencv-python:

    python3 -m pip install opencv-python

Clone Repository:

    git clone https://github.com/m2dsupsdlclass/lectures-labs


# Misc

## Draw neural net

[Draw neural net](http://alexlenail.me/NN-SVG/index.html)

