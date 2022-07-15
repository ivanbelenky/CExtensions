# Implementations

- `list`: implementation of a linked list. The idea behind this was to follow [this](https://docs.python.org/3/extending/extending.html) in order to learn how to define a module's `datatype` to be exported to `python`. It has almost no functionality beyond
  - end inserting
  - creation/elimination

- `ufuncs`: implementation of a universal function. The idea behind this was to follow [this](https://numpy.org/doc/stable/user/c-info.ufunc-tutorial.html) in order to learn how to define universal functions. In this manner you can leverage easily to have this functionality
  - broadcasting
  - N-dimensional looping
  - automatic type-conversions with minimal memory usage
  - optional output arrays

- `lreg`: Naive linear regression implementation. Minimizes quadratic distance. Completely algebraic definition. It is pretty sweet since it is ~30% faster than the replicated steps under raw numpy implementation. It is a good example of how to use `numpy's C API` and learn about reference counting.

- `numpy`: this was borrowed from [here](https://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html) but it proved to be useless. Deprecated usage of some functions. It provided minor insights. I would not discard it as useful material just yet.
