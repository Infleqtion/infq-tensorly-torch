Factorized tensors
==================

The core concept in TensorLy-Torch is that of *factorized tensors*.
We provide a :class:`~tltorch.FactorizedTensor` class that can be used just like any `PyTorch.Tensor` but 
provides all tensor factorization through one, simple API.


Creating factorized tensors
---------------------------

You can create a new factorized tensor easily:

The signature is: 

.. code-block:: python

   factorized_tensor = FactorizedTensor.new(shape, rank, factorization)

For instance, to create a tensor in Tucker form, that has half the parameters of a dense (non-factorized) tensor of the same shape, you would simply write:

.. code-block:: python

   tucker_tensor = FactorizedTensor.new(shape, rank=0.5, factorization='tucker')

Since TensorLy-Torch builds on top of TensorLy, it also comes with tensor decomposition out-of-the-box.
To initialize a factorized tensor in CP (Canonical-Polyadic) form, also known as Parafac, or Kruskal tensor,
with 1/10th of the parameters, you can simply write:

.. code-block:: python

   cp_tensor = FactorizedTensor.new(dense_tensor, rank=0.1, factorization='CP')


Manipulating factorized tensors
-------------------------------

The first thing you want to do, if you created a new tensor from scratch (by using the ``new`` method), is to initialize it, 
e.g. so that the element of the reconstruction approximately follow a Gaussian distribution:

.. code-block:: python

   cp_tensor.normal_(mean=0, std=0.02)

You can even use PyTorch's functions! This works:

.. code-block:: python

    from torch.nn import init

    init.kaiming_normal(cp_tensor)

Finally, you can index tensors directly in factorized form, which will return another factorized tensor, whenever possible!

>>> cp_tensor[:2, :2]
CPTensor(shape=(2, 2, 2), rank=2)

If not possible, a dense tensor will be returned:


>>> cp_tensor[2, 3, 1]
tensor(0.0250, grad_fn=<SumBackward0>)


Note how, above, indexing tracks gradients as well!

Tensorized tensors
==================

In addition to tensor in factorized forms, TensorLy-Torch provides out-of-the-box for **Tensorized** tensors. 
The most common case is that of tensorized matrices, where a matrix is first *tensorized*, i.e. reshaped into 
a higher-order tensor which is then decomposed and stored in factorized form.

A commonly used tensorized tensor is the tensor-train matrix (also known as Matrix-Product Operator in quantum physics),
or, in general, Block-TT.

Creation
--------

You can create one in TensorLy-Torch, from a matrix, just as easily as a regular tensor, using the :class:`tltorch.TensorizedTensor` class,
with the following signature:

.. code-block:: python

    TensorizedTensor.from_matrix(matrix, tensorized_row_shape, tensorized_column_shape, rank)

where tensorized_row_shape and tensorized_column_shape indicate the shape to which to tensorize the row and column size of the given matrix.
For instance, if you have a matrix of size 16x21, you could use tensorized_row_shape=(4, 4) and tensorized_column_shape=(3, 7).


In general, you can tensorize any tensor, not just matrices, even with batched modes (dimensions)!

.. code-block:: python

   tensorized_tensor = TensorizedTensor.new(tensorized_shape, rank, factorization)


``tensorized_shape`` is a nested tuple, in which an int represents a batched mode, and a tuple a tensorized mode.

For instance, a batch of 5 matrices of size 16x21 could be tensorized into 
a batch of 5 tensorized matrices of size (4x4)x(3x7), in the BlockTT form. In code, you would do this using

.. code-block:: python

   tensorized_tensor = TensorizedTensor.from_tensor(tensor, (5, (4, 4), (3, 7)), rank=0.7, factorization='BlockTT')

You can of course tensorize any size tensors, e.g. a batch of 5 matrices of size 8x27 can be tensorized into:

>>> ftt = tltorch.TensorizedTensor.new((5, (2, 2, 2), (3, 3, 3)), rank=0.5, factorization='BlockTT')

This returns a tensorized tensor, stored in decomposed form:
>>> ftt
BlockTT(shape=[5, 8, 27], tensorized_shape=(5, (2, 2, 2), (3, 3, 3)), rank=[1, 20, 20, 1])

Manipulation
-------------

As for factorized tensors, you can directly index them:

>>> ftt[2]
BlockTT(shape=[8, 27], tensorized_shape=[(2, 2, 2), (3, 3, 3)], rank=[1, 20, 20, 1])

>>> ftt[0, :2, :2]
tensor([[-0.0009,  0.0004],
        [ 0.0007,  0.0003]], grad_fn=<SqueezeBackward0>)

Again, notice that gradients are tracked and all operations on factorized and tensorized tensors are back-propagatable!

TT weight-storage targets
-------------------------

For TT and BlockTT, a positive floating-point ``rank`` requests a fraction of
original weight-tensor elements. ``'same'`` means ``1.0``. The integer parameter
budget is rounded down. Biases and unchanged layers are outside this budget;
complex weights are counted in complex elements. BlockTT uses merged physical
modes for its cores but counts source batch dimensions only once in the budget.

Fractional requests now select ranks that respect unfolding dimensions and both
neighboring-rank constraints before allocation or TT-SVD. Previously, TensorLy's
fractional heuristic could select oversized ranks that decomposition clipped,
leaving much of the intended storage unused. Fractional requests can therefore
produce different ranks and numerical approximations than earlier releases.
Explicit integer and tuple/list requests retain their rank-cap semantics,
including dimension-based clipping during decomposition.

Selection starts with the largest fitting capped common-rank profile, found by
binary search. Up to eight refinement passes consider increasing one bond or
reducing one bond by one while increasing another. Counts must strictly improve;
equal-count candidates prefer the least total absolute change from the starting
profile, then lexicographic order. This is a bounded storage heuristic, not a
globally optimal rank search or an accuracy/latency optimizer.

Budgets below the rank-one representation raise an error. Positive targets above
``1.0`` are supported; selection saturates when the largest feasible representation
fits within the budget. Integer steps and the bounded search can leave unused
budget. Such gaps are not necessarily mathematically unavoidable, and exact
attainment is not promised. Inspect the actual ranks and sum the core sizes to
obtain achieved storage; subtract that count from the floored budget for the gap.
