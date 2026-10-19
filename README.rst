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

Using uv for this checkout and its notebooks
-------------------------------------------

This checkout was initialized with ``uv init --bare``. TensorLy-Torch is a
standard Python package, not a special uv format. ``setup.py`` retains the
library's original metadata and requirements. ``pyproject.toml`` declares the
setuptools build backend and local dependency groups; ``uv.lock`` records their
resolved versions. The default ``dev`` group installs this checkout in editable
mode, so library edits are visible without reinstalling.

From the repository root, choose the environment you need::

    uv sync --locked
    uv sync --locked --group notebooks
    uv sync --locked --group notebooks --group opt-einsum
    uv sync --locked --group notebooks --group plot-styles

The first command installs the library, pytest, coverage support, and the tested
CPU numerical stack. The second adds JupyterLab, ipykernel, Matplotlib, and
TorchVision for notebooks. The third enables the opt_einsum comparison. The
fourth adds the optional Infleqtion plotting styles. Combine the group flags to
use both optional packages. ``uv sync`` removes packages from groups you do not
select; include the groups you want to keep each time.

Only ``dev`` is enabled by default. ``plot-styles`` installs from
``https://github.com/Infleqtion/matplotlib-styles`` at the revision in ``uv.lock``;
you need read access to that repository and Git authentication if required.
Without it, the main notebook uses ordinary Matplotlib styling. These groups
are local dependencies and are not published as library requirements.

``.python-version`` selects Python 3.12, and the ``dev`` and ``notebooks`` groups
are restricted to that Python series. You can select it explicitly with
``uv sync --python 3.12 --locked``. This CPU environment pins PyTorch 2.5.1,
TorchVision 0.20.1, TensorLy 0.9.0, NumPy 2.5.2, SciPy 1.17.1, and Matplotlib
3.10.8. TorchVision and Matplotlib are only installed with ``notebooks``.
These are tested environment versions, not new minimum versions for library
users. Other Python and dependency versions have not been validated here.

There is deliberately no ``[project]`` metadata table: the ``dev`` group refers
to this directory as an editable dependency, leaving packaging in ``setup.py``.
uv may report that it is inferring ``requires-python >=3.12`` for the workspace;
that applies to its environment lock, not the built library's metadata.

Start Jupyter or run tests with the relevant groups::

    uv run --locked --group notebooks jupyter lab
    uv run --locked --group notebooks --group opt-einsum --group plot-styles jupyter lab
    uv run --locked pytest

In VS Code's notebook kernel picker, choose **Python Environments** and select
this checkout's ``.venv/bin/python``. In JupyterLab, select this environment's
kernel rather than a historical kernel saved in notebook metadata. To give it
a recognizable name, run::

    uv run --locked --group notebooks python -m ipykernel install --sys-prefix --name tltorch-uv --display-name "TensorLy-Torch (uv)"

Then select **TensorLy-Torch (uv)**. This registers a kernel inside ``.venv``;
it does not replace kernels in other environments. Add environment dependencies
with ``uv add --group dev PACKAGE_NAME`` or ``uv add --group notebooks PACKAGE_NAME``.
Library runtime requirements remain in ``setup.py``. Keep ``.venv`` and build
artifacts out of Git.

The main notebook and CLI explicitly disable PyTorch's opt_einsum integration
in their shared runtime setup, even when the package is installed. New run
configurations record its availability, enabled flag, strategy, and the state
of TensorLy's separate plugin, which stays inactive. The opt_einsum comparison
still selects disabled and enabled execution explicitly; the compile comparison
keeps it disabled. Installing dependencies does not update historical results.

The model-compression notebooks still need cached weights, and the main accuracy
notebook also needs ImageNetV2. Follow ``examples/model_compression/README.md`` for
the data paths and explicit downloads. Choose new result directories before
rerunning notebooks; their saved directories already exist. The torch.compile
comparison also needs a working C++ compiler. Existing requirements files and
saved environments remain records of the earlier runs and are not changed by
this uv setup.

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
  




