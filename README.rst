.. image:: https://badge.fury.io/py/tensorly-torch.svg
    :target: https://badge.fury.io/py/tensorly-torch


==============
TensorLy-Torch
==============

TensorLy-Torch is a Python library for deep tensor networks that
builds on top of `TensorLy <https://github.com/tensorly/tensorly/>`_
and `PyTorch <https://pytorch.org/>`_.
It allows to easily leverage tensor methods in a deep learning setting and comes with all batteries included.

- **Website:** http://tensorly.org/torch/
- **Source-code:**  https://github.com/tensorly/torch


With TensorLy-Torch, you can easily: 

- **Tensor Factorizations**: decomposing, manipulating and initializing tensor decompositions can be tricky. We take care of it all, in a convenient, unified API.
- **Leverage structure in your data**: with tensor layers, you can easily leverage the structure in your data, through Tensor Regression Layers, Factorized Convolutions, etc
- **Built-in tensor layers**: all you have to do is import tensorly torch and include the layers we provide directly within your PyTorch models!
- **Tensor hooks**: you can easily augment your architectures with our built-in Tensor Hooks. Robustify your network with Tensor Dropout and automatically select the rank end-to-end with L1 Regularization!
- **All the methods available**: we are always adding more methods to make it easy to compare between the performance of various deep tensor based methods!

Deep Tensorized Learning
========================

Tensor methods generalize matrix algebraic operations to higher-orders. Deep neural networks typically map between higher-order tensors. 
In fact, it is the ability of deep convolutional neural networks to preserve and leverage local structure that, along with large datasets and efficient hardware, made the current levels of performance possible.
Tensor methods allow to further leverage and preserve that structure, for individual layers or whole networks. 

.. image:: ./doc/_static/tensorly-torch-pyramid.png

TensorLy is a Python library that aims at making tensor learning simple and accessible.
It provides a high-level API for tensor methods, including core tensor operations, tensor decomposition and regression. 
It has a flexible backend that allows running operations seamlessly using NumPy, PyTorch, TensorFlow, JAX, MXNet and CuPy.
 
**TensorLy-Torch** is a PyTorch only library that builds on top of TensorLy and provides out-of-the-box tensor layers.

Improve your neural networks with tensor methods
------------------------------------------------

Tensor methods generalize matrix algebraic operations to higher-orders. Deep neural networks typically map between higher-order tensors. 
In fact, it is the ability of deep convolutional neural networks to preserve and leverage local structure that, along with large datasets and efficient hardware, made the current levels of performance possible.
Tensor methods allow to further leverage and preserve that structure, for individual layers or whole networks. 

In TensorLy-Torch, we provide convenient layers that do all the heavy lifting for you 
and provide the benefits tensor based layers wrapped in a nice, well documented and tested API.

For instance, convolution layers of any order (2D, 3D or more), can be efficiently parametrized
using tensor decomposition. Using a CP decomposition results in a separable convolution
and you can replace your original convolution with a series of small efficient ones: 

.. image:: ./doc/_static/cp-conv.png 

These can be easily perform with FactorizedConv in TensorLy-Torch.
We also have Tucker convolutions and new tensor-train convolutions!
We also implement various other methods such as tensor regression and contraction layers, 
tensorized linear layers, tensor dropout and more!


Installing TensorLy-Torch
=========================

Using uv for development and the ImageNet experiment
----------------------------------------------------

This checkout was initialized with ``uv init --bare``. ``setup.py`` retains the
library's original metadata and requirements. ``pyproject.toml`` defines the
setuptools backend and development groups; ``uv.lock`` records their versions.
The default ``dev`` group installs this checkout in editable mode. Python 3.12
is selected by ``.python-version``. From the repository root::

    uv sync --locked

To include the experiment, notebook, and optional plotting tools while retaining
other installed tools::

    uv sync --locked --inexact --group experiments --group notebooks --group opt-einsum --group plot-styles

Include the groups you want to manage each time you synchronize. ``plot-styles``
requires read access to the locked Infleqtion Git repository. These are environment
dependencies, not new published library requirements. Packaging remains in
``setup.py``; there is deliberately no duplicate ``[project]`` metadata table.

The ImageNet workflow lives in ``experiments/imagenet-compression/``. Its
`README <experiments/imagenet-compression/README.md>`_ explains data preparation,
the single runner, configurations, local CPU/CUDA setup, VSCode/Jupyter selection,
tracking, checkpoints, notebooks, and historical records. The CUDA environment
also runs CPU experiments without reinstalling packages. Existing measurements
retain their original recorded environments.

Using this library in another uv project
---------------------------------------

For a new project, start with ``uv init --bare``. Then use the local checkout::

    uv add --editable /path/to/infq-tensorly-torch-model-compression
    uv add tensorly torch packaging

The original library metadata declares only NumPy and SciPy. Install TensorLy,
PyTorch, and ``packaging`` explicitly in the consuming project as well; the
TensorLy PyTorch backend uses ``packaging``. Choose versions appropriate to that
project. The local development groups above are not inherited by consumers.

Editable installation follows local source changes, including uncommitted ones.
Omit ``--editable`` to install a copy. Once the desired changes are committed and
pushed, another project can instead use this fork's Git branch::

    uv add git+https://github.com/Infleqtion/infq-tensorly-torch.git --branch rafael/model-compression

Use ``--rev COMMIT_HASH`` instead of ``--branch`` to choose a specific commit.
``uv add tensorly-torch`` selects the public package index release, which does not
include this fork's unpublished changes. No publication to PyPI is required for
local-path or Git installation.

The CPU wheel index is configured for this checkout only. Another project
chooses its own PyTorch build. To use CPU wheels there, add the following to
that project's ``pyproject.toml`` before adding PyTorch::

    [tool.uv.sources]
    torch = {index = "pytorch-cpu"}

    [[tool.uv.index]]
    name = "pytorch-cpu"
    url = "https://download.pytorch.org/whl/cpu"
    explicit = true

If those tables already exist, add the entries to them instead of duplicating
the tables. See the `uv dependency guide
<https://docs.astral.sh/uv/concepts/projects/dependencies/>`_ and
`PyTorch integration guide
<https://docs.astral.sh/uv/guides/integration/pytorch/>`_ for other sources and builds.

Through pip
-----------

.. code:: 

   pip install tensorly-torch
   
   
From source
-----------

.. code::

  git clone https://github.com/tensorly/torch
  cd torch
  pip install -e .
  




