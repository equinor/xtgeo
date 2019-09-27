#include "logger.h"

static PyObject *logger;

void logger_init(const char *fname)
{
    static PyObject *logging = NULL;

    // import logging module on demand
    if (logging == NULL){
        logging = PyImport_ImportModuleNoBlock("logging");
        if (logging == NULL) {
            PyErr_SetString(PyExc_ImportError,
                "Could not import module 'logging'");
        }

    }
    logger = PyObject_CallMethod(logging, "getLogger", "s", fname);
}


void logger_info(char *msg)
{
    static PyObject *string = NULL;

    string = Py_BuildValue("s", msg);

    PyObject_CallMethod(logger, "info", "O", string);
    Py_DECREF(string);
}

void logger_warn(char *msg)
{
    static PyObject *string = NULL;

    string = Py_BuildValue("s", msg);

    PyObject_CallMethod(logger, "warning", "O", string);
    Py_DECREF(string);
}
