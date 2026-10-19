=============
API reference
=============

:mod:`tltorch`: Tensorized Deep Neural Networks

.. automodule:: tltorch
    :no-members:
    :no-inherited-members:

.. _factorized_tensor_ref:

Factorized Tensors
==================

TensorLy-Torch builds on top of TensorLy and provides out of the box PyTorch layers for tensor based operations.
The core of this is the concept of factorized tensors, which factorize our layers, instead of regular, dense PyTorch tensors.

You can create any factorized tensor through the main class using:

.. autosummary::
    :toctree: generated
    :template: class.rst

    FactorizedTensor

You can create a tensor of any form using ``FactorizedTensor.new(shape, rank, factorization)``, where factorization can be `Dense`, `CP`, `Tucker` or `TT`.
Note that if you use ``factorization = 'dense'`` you are just creating a regular, unfactorized tensor. 
This allows to manipulate any tensor, factorized or not, with a simple, unified interface.

Alternatively, you can also directly create a specific subclass:

.. autosummary::
    :toctree: generated
    :template: class.rst

    DenseTensor
    CPTensor
    TuckerTensor
    TTTensor

.. _factorized_matrix_ref:

Tensorized Matrices
===================

In TensorLy-Torch , you can also represent matrices in *tensorized* form, as low-rank tensors.
 Just as for factorized tensor, you can create a tensorized matrix through the main class using:

.. autosummary::
    :toctree: generated
    :template: class.rst

    TensorizedTensor

You can create a tensor of any form using ``TensorizedTensor.new(tensorized_shape, rank, factorization)``, where factorization can be `Dense`, `CP`, `Tucker` or `BlockTT`.

You can also explicitly create the type of tensor you want using the following classes:

.. autosummary::
    :toctree: generated
    :template: class.rst

    DenseTensorized
    TensorizedTensor
    CPTensorized
    BlockTT

.. _complex_ref:

Complex Tensors
===============

In theory, you can simply specify ``dtype=torch.cfloat`` in the creation of any of the tensors of tensorized matrices above, to automatically create a complex valued tensor.
However, in practice, there are many issues in complex support. Distributed Data Parallelism in particular, is not supported.

In TensorLy-Torch, we propose a convenient and transparent way around this: simply use ``ComplexTensor`` instead. 
This will store the factors of the decomposition in real form (by explicitly storing the real and imaginary parts) 
but will transparently return you a complex valued tensor or reconstruction.

.. autosummary::
    :toctree: generated
    :template: class.rst

    ComplexDenseTensor
    ComplexCPTensor
    ComplexTuckerTensor
    ComplexTTTensor


    ComplexDenseTensorized
    ComplexTuckerTensorized
    ComplexCPTensorized
    ComplexBlockTT


You can also transparently instanciate any of these using directly the main classes, ``TensorizedTensor`` or ``FactorizedTensor`` and specifying 
``factorization="ComplexCP"`` or in general ``ComplexFactorization`` with `Factorization` any of the supported decompositions.


.. _init_ref:

Initialization
==============

.. automodule:: tltorch.factorized_tensors
    :no-members:
    :no-inherited-members:

Initialization is particularly important in the context of deep learning. 
We provide convenient functions to directly initialize factorized tensor (i.e. their factors)
such that their reconstruction follows approximately a centered Gaussian distribution. 

.. currentmodule:: tltorch.factorized_tensors.init

.. autosummary::
    :toctree: generated
    :template: function.rst

    tensor_init
    cp_init
    tucker_init
    tt_init
    block_tt_init

.. _trl_ref:

Tensor Regression Layers
========================

.. automodule:: tltorch.factorized_layers
    :no-members:
    :no-inherited-members:

.. currentmodule:: tltorch.factorized_layers

.. autosummary::
    :toctree: generated
    :template: class.rst

    TRL

.. _tcl_ref:

Tensor Contraction Layers
=========================

.. autosummary::
    :toctree: generated
    :template: class.rst

    TCL

.. _factorized_linear_ref:

Factorized Linear Layers
========================

.. autosummary::
    :toctree: generated
    :template: class.rst

    FactorizedLinear

.. _factorized_conv_ref:

Factorized Convolutions
=======================

General N-Dimensional convolutions in Factorized forms

.. autosummary::
    :toctree: generated
    :template: class.rst

    FactorizedConv

.. _tensor_dropout_ref:

Factorized Embeddings
=====================

A drop-in replacement for PyTorch's embeddings but using an efficient tensor parametrization that never reconstructs the full table. 

.. autosummary::
    :toctree: generated
    :template: class.rst

    FactorizedEmbedding

.. _tensor_dropout_ref:

Tensor Dropout
==============

.. currentmodule:: tltorch.tensor_hooks

.. automodule:: tltorch.tensor_hooks
    :no-members:
    :no-inherited-members:

These functions allow you to easily add or remove tensor dropout from tensor layers.


.. autosummary::
    :toctree: generated
    :template: function.rst

    tensor_dropout
    remove_tensor_dropout


You can also use the class API below but unless you have a particular use for the classes, you should use the convenient functions provided instead.

.. autosummary::
    :toctree: generated
    :template: class.rst

    TensorDropout

.. _tensor_lasso_ref:

L1 Regularization
=================

L1 Regularization on tensor modules. 

.. currentmodule:: tltorch.tensor_hooks

.. autosummary::
    :toctree: generated
    :template: function.rst

    tensor_lasso
    remove_tensor_lasso

Utilities
=========

Utility functions

.. currentmodule:: tltorch.utils

.. autosummary::
    :toctree: generated
    :template: function.rst

    get_tensorized_shape

Model compression
=================

``tltorch.compress_model`` copies a model and replaces selected trained layers
using the existing single-layer converters. Linear layers support CP, Tucker,
and BlockTT; Conv2d layers support CP, Tucker, and TT. Both factorized and
reconstructed execution are available::

    from tltorch import compress_model

    compressed, report = compress_model(model, {
        "classifier": {
            "factorization": "CP",
            "rank": 8,
            "in_tensorized_features": (4, 4),
            "out_tensorized_features": (2, 5),
            "decomposition_kwargs": {"random_state": 7, "n_iter_max": 100},
        },
    })

This example assumes a classifier with 16 inputs and 10 outputs. Omit both
mode lists to use the existing automatic tensorization. The lists must have
equal lengths: this scope restriction avoids a known Tucker factorized-forward
problem, although CP itself handles unequal lengths. Tucker linear rank tuples
list all output modes, then all input modes. CP requires a scalar rank.

Integer ranks specify sizes; floating-point ranks and ``"same"`` request
approximate weight-storage targets. They do not guarantee savings. Tucker can
store more parameters than the original weight for some targets. Use
``max_parameters`` to enforce a limit on the actual replacement, including bias
and frozen parameters. The report counts CP coefficients and Tucker cores as
well as factor matrices.

CP uses random initialization with seed 0, up to 100 fitting iterations, tolerance
``1e-8``, and ``l2_reg=1e-5``. Random initialization starts a fit to the trained
weights; it never substitutes unfitted random weights. This wrapper rejects CP
SVD initialization because the installed TensorLy path can allocate excessive
temporary tensors. Random initialization avoids that path, but large ranks can
still be expensive. There are no hidden rank caps or retries.

CP and Tucker accept ``init``, ``n_iter_max``, ``tol``, and ``random_state`` in
``decomposition_kwargs``; only CP accepts ``l2_reg``. Tucker retains SVD
initialization, up to 100 iterations, tolerance ``1e-4``, and ``random_state=None`` by
default, and also supports random initialization. Iteration limits must be
positive integers; tolerance and regularization must be finite and nonnegative.
Explicit seeds must be integers from 0 through ``2**32 - 1``. Seeds pass directly
to TensorLy without changing global NumPy state; Tucker's ``None`` keeps its
unseeded behavior. Other options, including ``return_errors``, are rejected.
TT and BlockTT do not accept decomposition controls.

Each replacement report includes its effective ``decomposition_kwargs`` (an
empty mapping for TT and BlockTT). Shared aliases compare these normalized
settings, so omitted controls and explicit defaults agree. ``actual_rank`` is
an integer for CP and a tuple for Tucker, TT, or BlockTT, describing the stored
representation after fitting, including any rank reduction. Weight approximation
error does not establish task accuracy.

.. autofunction:: tltorch.compress_model
