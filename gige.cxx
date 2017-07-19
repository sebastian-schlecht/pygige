#include <methods.hpp>


void
py_close(PyObject* deviceCapsule) {
    gige::IDevice* ptr = (gige::IDevice*) PyCapsule_GetPointer(deviceCapsule, "Device");
    close(*ptr);
    delete [] ptr;
}

static PyObject* 
py_setup(PyObject *self, PyObject *args) {
    PyObject *pObj = NULL;
    gige::IDevice device = setup();
    gige::IDevice* ptr = new gige::IDevice(device);
    if(device != NULL) {
       return PyCapsule_New(ptr, "Device", py_close); 
    }
    else {
        return Py_None;
    }
    
}

static PyObject*
py_get_frame(PyObject *self, PyObject *args) {
    PyObject *deviceCapsule = NULL;
    float* timeout;
    if(!PyArg_ParseTuple(args, "Of", &deviceCapsule, &timeout)) {
        return NULL;
    }
    if(deviceCapsule == Py_None) {
        return Py_None;
    }
    gige::IDevice* devicePtr = (gige::IDevice*) PyCapsule_GetPointer(deviceCapsule, "Device");
    Frame f = getFrame(*devicePtr, *timeout);
    npy_intp dims[] = {f.y, f.x};
    PyArrayObject *array = (PyArrayObject*) PyArray_SimpleNewFromData(2,
                                      dims,
                                      NPY_UINT8,
                                      f.data);
    return (PyObject*)array;
};

/* Define functions in module */
static PyMethodDef GigeMethods[] = {
    {"setup", py_setup, METH_VARARGS, "Setup the device"},
    {"getFrame", py_get_frame, METH_VARARGS, "Obtain a frame"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Module initialization */
PyMODINIT_FUNC
initgige(void)
{
    //import_array();
    (void) Py_InitModule("gige", GigeMethods);
}