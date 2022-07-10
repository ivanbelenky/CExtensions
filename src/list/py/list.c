#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <Python.h>
#include "structmember.h"
#define PY_SSIZE_T_CLEAN


/*       Node       */

typedef struct{
    PyObject_HEAD
    PyObject *next;
    PyObject *data;
} Node;



static PyObject * 
Node_new(PyTypeObject *type, PyObject *args, PyObject *kwds){
    Node *self;
    self = (Node *)type->tp_alloc(type, 0);
    if (self == NULL){
        return NULL;
    }
    self->next = Py_None;
    self->data = Py_None;
        
    return (PyObject *)self;
};

static int 
Node_init(Node *self, PyObject *args, PyObject *kwds){
    PyObject *data = NULL, *tmp;
    /* The second argument for UnpackTuple is used when raising exceptions */
    if(!PyArg_UnpackTuple(args, "__init__", 0, 1, &data)){
        return -1;
    }
    /* Why doing it this way?  
    check 2.2 subsection on https://docs.python.org/3/extending/newtypes_tutorial.html*/
    if (data){
        tmp = self->data;
        Py_INCREF(data);
        self->data = data;
        Py_DECREF(tmp);
    }

    return 0;
};

static int 
Node_clear_refs(Node *self){
    Py_CLEAR(self->next);
    Py_CLEAR(self->data);
    return 0;
} 

static void
Node_dealloc(Node *self){
    PyObject *obj_self = (PyObject*)self;
    Node_clear_refs(self);
    Py_TYPE(obj_self)->tp_free(obj_self);
};


static PyMemberDef 
NodeMembers[] = {
    { "data", T_OBJECT_EX, offsetof(Node, data), 0, "data" },
    { "next", T_OBJECT_EX, offsetof(Node, next), READONLY, "next node" },
    { NULL },   /* sentinel */
};

static PyTypeObject NodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Linked Listed Node ",
    .tp_doc = "Nothing but an Object holder with a reference to the next one",
    .tp_itemsize = 0,
    .tp_basicsize = sizeof(Node),
    .tp_dealloc = (destructor)Node_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Node_new,
    .tp_init = (initproc)Node_init,
    .tp_members = NodeMembers,  
    .tp_clear = (inquiry)Node_clear_refs,
};



/*       Linked List       */


typedef struct{
    PyObject_HEAD
    PyObject *first;
    unsigned int size;
} Llist;


static PyObject * 
Llist_new(PyTypeObject *type, PyObject *args, PyObject *kwds){
    Llist *self;
    self = (Llist *)type->tp_alloc(type, 0);
    if (self == NULL){
        return NULL;
    }
    self->first = Py_None;
    self->size = 0;
        
    return (PyObject *)self;
};

static int
Llist_insert_sequence(Llist *self, PyObject *sequence){    
    Py_ssize_t i;
    Py_ssize_t sequence_len;

    sequence_len = PySequence_Length(sequence);
    if (sequence_len == -1){
        PyErr_SetString(PyExc_ValueError, "Invalid Sequence");
    }

    for (i=0; i < sequence_len; i++){
        PyObject *item;
        PyObject *new_node;

        item = PySequence_GetItem(sequence, i);
        if (item == NULL){
            PyErr_SetString(PyExc_ValueError, "Failed to get element from sequence");
            return 0;
        }
        
        new_node = Node_new(&NodeType, NULL, NULL);
        if (new_node == NULL){
            Py_DECREF(item);
            return 0;
        }
        
        ((Node*)new_node)->data = item;
        if (i==0){
            self->first = new_node;
        }        
        else {
            Node *tmp = (Node*)self->first;
            while(tmp->next != Py_None){
                tmp = (Node*)tmp->next;
            }
            tmp->next = new_node;
        }

        self->size += 1;
        Py_DECREF(item);
    }
    
    return 1;
};

static int 
Llist_init(Llist *self, PyObject *args, PyObject *kwds){
    PyObject *sequence = NULL;
    if (!PyArg_UnpackTuple(args, "__init__", 0, 1, &sequence))
        return -1;
    
    if (sequence==NULL){
        return 0;
    }

    if (!PySequence_Check(sequence)){
        PyErr_SetString(PyExc_TypeError, "Argument must be a sequence");
        return -1;
    }
    
    return Llist_insert_sequence(self, sequence) ? 0 : -1;
};


static int
Llist_clear_refs(Llist *self){
    PyObject *node = self->first;

    self->first = NULL;
    
    if (node != NULL){
        while (node != Py_None){
            PyObject *next_node = ((Node*)node)->next;
            ((Node*)node)->next = Py_None;        
            Py_DECREF(node);
            node = next_node;
        }
    }
    return 0;
};

static void 
Llist_dealloc(Llist *self){
    PyObject* obj_self = (PyObject*)self;
    Llist_clear_refs(self);
    obj_self->ob_type->tp_free(obj_self);
};


static PyMemberDef LlistMembers[] ={
    { "first", T_OBJECT_EX, offsetof(Llist, first), READONLY,
      "First node" },
    { "size", T_INT, offsetof(Llist, size), READONLY,
      "size" },
    { NULL },   /* sentinel */
};

static PyTypeObject LlistType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Linked List",
    .tp_doc = "A linked list",
    .tp_itemsize = 0,
    .tp_basicsize = sizeof(Llist),
    .tp_dealloc = (destructor)Llist_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Llist_new,
    .tp_init = (initproc)Llist_init,
    .tp_clear = (inquiry)Llist_clear_refs,
    .tp_members = LlistMembers,
};


/*       Module       */


static PyModuleDef llist = {
    PyModuleDef_HEAD_INIT,
    .m_name = "llist",
    .m_doc = "Linked List module.",
    .m_size = -1,
};


PyMODINIT_FUNC
PyInit_llist(void){
    PyObject *m;
    if (PyType_Ready(&NodeType) < 0)
        return NULL; 
    
    if (PyType_Ready(&LlistType) < 0)
        return NULL;

    m = PyModule_Create(&llist);
    if (m == NULL)
        return NULL;

    Py_INCREF(&NodeType);
    if (PyModule_AddObject(m, "Node", (PyObject *) &NodeType) < 0) {
        Py_DECREF(&NodeType);
        Py_DECREF(m);
        return NULL;
    }
    
    Py_INCREF(&LlistType);
    if (PyModule_AddObject(m, "Llist", (PyObject *) &LlistType) < 0) {
        Py_DECREF(&LlistType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
};


