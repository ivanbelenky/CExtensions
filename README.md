# Writing C, Python/C API 

The idea behind this repo is to start building core knowledge on how to extend Python with C or C++. So dont hesitate to be verbose on code comments or README.


## Guidelines
### `src`

Please follow the guideline below when committing examples of C extensions.
1. Create new folder under `src` named accordingly to the example, i.e. `list`
2. Create 1 or 2 subfolders when adequate, `py`, `c`. If this code is done from scratch using PYTHON/C API no need to create the `c` folder. If you are rewriting some `C` sources as an extension, please paste those on `c` so we can use that as reference when reading.
3. Create `setup.py`
4. Provide additionals Readme if suitable

### `tests`

One file with the same name as the example you are posting. 


## Current

- `llist`: simple rewriting of a linked list interface written by me a long time ago.
- `numpy`: code from <a href="https://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html"> here</a>. Useful whenever the time comes to do some more fancy shit using `arrayobjects`.


## Resources to Start

1. As a first step I would suggest <a href="https://docs.python.org/3/extending/extending.html"></a>
2. More on Reference Counting mainly <a href="https://pythonextensionpatterns.readthedocs.io/en/latest/refcount.html">extensionpatterns</a>
3. `numpy` <a href="https://numpy.org/doc/stable/user/c-info.html">np</a>

