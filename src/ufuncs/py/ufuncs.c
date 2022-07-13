#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include <math.h>

#define MAXITERTOL 10
#define MAXITER 3

static PyMethodDef UfuncMethods[] = {
    {NULL, NULL, 0, NULL}
};

static void double_sqrt_tol(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    double tmp;
    double root=0, last_root=-1;
    int iter=0;

    for (i = 0; i < n; i++) {
        tmp = *(double *)in;
        while (iter <= MAXITERTOL ) {
            if (iter%3 == 0) {
                if (fabs(root-last_root) < 1e-6){
                    break;
                }
            }
            last_root = root;
            root = 1 + (tmp-1)/(1+root); 
            iter++;
        }       
        *((double *)out) = root;
        root = 0;
        last_root = -1;
        iter = 0;

        in += in_step;
        out += out_step;
    }
}

static void double_sqrt(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    double tmp;
    double root=0;
    int iter=0;

    for (i = 0; i < n; i++) {
        tmp = *(double *)in;
        while (iter <= MAXITER ) {
            root = 1 + (tmp-1)/(1+root); 
            iter++;
        }       
        *((double *)out) = root;
        root = 0;
        iter = 0;

        in += in_step;
        out += out_step;
    }
}


/* This a pointer to the above function */
PyUFuncGenericFunction sqrtfuncs[1] = {&double_sqrt};
PyUFuncGenericFunction sqrttolfuncs[1] = {&double_sqrt_tol};



/* These are the input and return dtypes of both ufuncs.*/
static char types[2] = {NPY_DOUBLE, NPY_DOUBLE};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    UfuncMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_npufunc(void)
{
    PyObject *m, *sqrt, *sqrttol, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    sqrt = PyUFunc_FromFuncAndData(sqrtfuncs, NULL, types, 1, 1, 1,
                                    PyUFunc_None, "sqrt",
                                    "sqrt_docstring", 0);

    sqrttol = PyUFunc_FromFuncAndData(sqrttolfuncs, NULL, types, 1, 1, 1,
                                PyUFunc_None, "sqrt2",
                                "sqrt_docstring", 0);


    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "sqrt", sqrt);
    Py_DECREF(sqrt);
    PyDict_SetItemString(d, "sqrt2", sqrttol);
    Py_DECREF(sqrttol);

    return m;
}
