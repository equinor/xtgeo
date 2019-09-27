#include "logger.h"

char* _concat(const char *s1, const char *s2)
{
    char *result = malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}




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
    char *fullname = _concat("<cxtgeo>:", fname);
    logger = PyObject_CallMethod(logging, "getLogger", "s", fullname);
    free(fullname);
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
