#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>

void
print_shape(PyObject *arr){
    int i;
    int ndim = PyArray_NDIM((PyArrayObject*)arr);
    
    printf("(");
    for (i=0; i<ndim-1; i++){
        printf("%ld, ", PyArray_DIM(arr, i));
    }
    printf("%ld)\n", PyArray_DIM(arr,ndim-1));
    return;
}

void
lazy_print(PyArrayObject *arr){
    FILE *fp = stdout;
    printf("\n[");
    PyArray_ToFile(arr, fp, ", ", NULL);
    printf("]\n");
    printf("\n");
}


static PyObject *
PyArray_2x2_Inverse(PyArrayObject *mat){
    int nd = 2;
    npy_intp const dims[2] = {2,2};
    PyObject *inv;
    double a, b, c, d, det;
    
    inv = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);

    a = *(double*)PyArray_GETPTR2(mat, 0, 0);
    b = *(double*)PyArray_GETPTR2(mat, 0, 1);
    c = *(double*)PyArray_GETPTR2(mat, 1, 0);
    d = *(double*)PyArray_GETPTR2(mat, 1, 1);

    det = a*d-b*c;

    *(double*)PyArray_GETPTR2(inv, 0, 0) = 1/det * d;
    *(double*)PyArray_GETPTR2(inv, 0, 1) = 1/det * -b; 
    *(double*)PyArray_GETPTR2(inv, 1, 0) = 1/det * -c;
    *(double*)PyArray_GETPTR2(inv, 1, 1) = 1/det * a;

    return inv;
}


static PyObject * 
transpose_1d(PyArrayObject* arr){
    PyObject *t_shape;
    PyObject *arr_t;

    if (PyArray_NDIM(arr) != 1 && PyArray_NDIM(arr) != 2 ){
        PyErr_SetString(PyExc_ValueError, "Only (n,), (n,1) and (1,n) shaped arrays are supported");
        return NULL;
    }
    

    /* if arr is shaped (n,) convert it to (1, n) -> transpose -> (n,1)*/
    if (PyArray_NDIM(arr) == 1){
        t_shape = Py_BuildValue("[ll]", PyArray_DIM(arr, 0), 1);
        arr_t = PyArray_Reshape(arr, t_shape);
        
        Py_DECREF(t_shape);
        return arr_t;
    }
    
    if (PyArray_NDIM(arr) == 2){
        /* if arr is shaped (n, 1) */
        if (PyArray_DIM(arr, 1) == 1){
            t_shape = Py_BuildValue("[ll]", 1, PyArray_DIM(arr, 0));
            arr_t = PyArray_Reshape(arr, t_shape);
            Py_DECREF(t_shape);
            return arr_t;
        }
        else if(PyArray_DIM(arr, 0) == 1){
            t_shape= Py_BuildValue("[ll]", PyArray_DIM(arr, 1), 1);
            arr_t = PyArray_Reshape(arr, t_shape);
            Py_DECREF(t_shape);
            return arr_t;
        }    
    }

    PyErr_SetString(PyExc_ValueError, "Only (n,), (n,1) and (1,n) shaped arrays are supported");
    return NULL;
}


static PyObject *
PyArray_ONES(int size){
    PyObject *arr_ones;
    int i;
    npy_intp const dims[1] = {size};
    arr_ones = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    for (i=0; i<size; i++){
        *(double*)PyArray_GETPTR1(arr_ones, i) = 1;
    }
    return arr_ones;
}



static PyArrayObject *
PyArray_1d_HSTACK(PyArrayObject *arr1, PyArrayObject *arr2){
    double **out;
    int nel1, nel2;
    int i;
    PyArrayObject *out_arr;

    if (PyArray_NDIM(arr1) == 1){
        arr1 = (PyArrayObject*)transpose_1d(arr1);
    }
    if (PyArray_NDIM(arr2) == 1){
        arr2 = (PyArrayObject*)transpose_1d(arr2);
    }

    if (PyArray_NDIM(arr1) != 2 || PyArray_NDIM(arr2) != 2){
        PyErr_SetString(PyExc_ValueError, "Only (n,1) and (1,n) shaped arrays are supported");
        return NULL;
    }

    if ((PyArray_DIM(arr1, 1) != 1) && (PyArray_DIM(arr1, 0) == 1)){
        arr1 = (PyArrayObject*)transpose_1d(arr1);
    }
    else if ((PyArray_DIM(arr1, 1) != 1) && (PyArray_DIM(arr1, 0) != 1)){
        PyErr_SetString(PyExc_ValueError, "Only (n,1) and (1,n) shaped arrays are supported");
        return NULL;
    }

    if ((PyArray_DIM(arr2, 1) != 1) && (PyArray_DIM(arr2, 0) == 1)){
        arr2 = (PyArrayObject*)transpose_1d(arr2);
    }
    else if ((PyArray_DIM(arr2, 1) != 1) && (PyArray_DIM(arr2, 0) != 1)){
        PyErr_SetString(PyExc_ValueError, "Only (n,1) and (1,n) shaped arrays are supported");
        return NULL;
    }    
    
    nel1 = PyArray_DIM(arr1, 0);
    nel2 = PyArray_DIM(arr2, 0);
    
    if(nel1 != nel2){
        PyErr_SetString(PyExc_ValueError, "Arrays must have the same number of elements");
        return NULL;
    } 

    npy_intp const dims[2] = {nel1, 2};
    out_arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    for (i=0; i<nel1; i++){
        *(double*)PyArray_GETPTR2(out_arr, i, 0) = *(double*)PyArray_GETPTR2(arr1, i, 0);
        *(double*)PyArray_GETPTR2(out_arr, i, 1) = *(double*)PyArray_GETPTR2(arr2, i, 0);
    }

    return out_arr;
}


static PyObject *
_naive_regression(PyObject *x, PyObject *y){
    PyObject  *x_t, *y_t, *ones, *ones_t, *b;
    PyObject *A, *At, *AtA, *AtA_inv;

    if ((x_t = transpose_1d((PyArrayObject*)x)) == NULL){
        return NULL;
    }
    if ((y_t = transpose_1d((PyArrayObject*)y)) == NULL){
        Py_DECREF(x_t);
        return NULL;
    }

    ones = PyArray_ONES(PyArray_DIM(x_t, 0));
    ones_t = transpose_1d((PyArrayObject*)ones);

    A = (PyArrayObject*)PyArray_1d_HSTACK((PyArrayObject*)x_t, (PyArrayObject*)ones_t);
    At = PyArray_Transpose(A, NULL);
    AtA = PyArray_MatrixProduct(At, A);
    AtA_inv = PyArray_2x2_Inverse(AtA);
    b = ((PyArrayObject*)PyArray_MatrixProduct(AtA_inv,(PyArrayObject*)PyArray_MatrixProduct(At, y_t)));

    Py_DECREF(x_t);
    Py_DECREF(y_t);
    Py_DECREF(ones);
    Py_DECREF(ones_t);
    Py_DECREF(A);
    Py_DECREF(At);
    Py_DECREF(AtA);
    Py_DECREF(AtA_inv);
    Py_INCREF(Py_None);
    return b;
}


static PyObject *
naive_linear_regression(PyObject *self, PyObject *args){
    PyObject *x, *y;
    PyObject *arr_x, *arr_y;
    PyObject *b;
    
    if (!PyArg_ParseTuple(args, "OO", &x, &y))
        return NULL;

    arr_x = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_IN_ARRAY);
    arr_y = PyArray_FROM_OTF(y, NPY_DOUBLE, NPY_IN_ARRAY);

    if (PyArray_NDIM(arr_x) != 1 || PyArray_NDIM(arr_y) != 1) {
        PyErr_SetString(PyExc_ValueError, "x and y must be 1-dimensional");
        goto fail;
    }

    b = _naive_regression(arr_x, arr_y);

    Py_XDECREF(arr_x);
    Py_XDECREF(arr_y);
    Py_INCREF(Py_None);
    return b;

    fail:
        Py_XDECREF(arr_x);
        Py_XDECREF(arr_y);
        Py_INCREF(Py_None);
        return Py_None;
}


static PyObject *
matmul(PyObject *dummy, PyObject *args){
    Py_INCREF(Py_None);
    return Py_None;
    
}

static PyMethodDef LregMethods[] = {
	{"naive", naive_linear_regression, METH_VARARGS, "Naive linear regression"},
    {"matmul", matmul, METH_VARARGS, "Matrix Multiplication"},
	{NULL, NULL, 0, NULL}     /* Sentinel - marks the end of this structure */
};


static struct PyModuleDef lreg = {
    PyModuleDef_HEAD_INIT,
    "Combinations", /* name of module */
    "usage: Combinations.uniqueCombinations(lstSortableItems, comboSize)\n", /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    LregMethods,
};

PyMODINIT_FUNC
PyInit_lreg(void){
    PyObject *m;
    if ((m=PyModule_Create(&lreg))==NULL)
        return NULL;
    
    import_array();

    return m;
}