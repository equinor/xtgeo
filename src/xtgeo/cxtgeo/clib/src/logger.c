#include "logger.h"

/* Logging levels: */

/* CRITICAL: 50 */
/* ERROR: 40 */
/* WARNING: 30 */
/* INFO: 20 */
/* DEBUG: 10 */
/* NOTSET: 0 */

static PyObject *logger;
static int logging_level;

void logger_init(const char *fname)
{
    static PyObject *logging = NULL;
    char fullname[50];

    // import logging module on demand
    if (logging == NULL){
        logging = PyImport_ImportModuleNoBlock("logging");
        if (logging == NULL) {
            PyErr_SetString(PyExc_ImportError,
                "Could not import module 'logging'");
        }

    }

    sprintf(fullname, "<cxtgeo>: %s", fname);

    logger = PyObject_CallMethod(logging, "getLogger", "s", fullname);

    /* Get the effective logging level */
    PyObject *meth = PyObject_GetAttrString(logger, "getEffectiveLevel");
    PyObject *level = PyObject_CallFunctionObjArgs(meth, NULL);

    Py_DECREF(meth);

    if(level == NULL) {
        logging_level = 50;
    }
    else{
        logging_level = PyLong_AsLong(level);
    }
    if(PyErr_Occurred()){
        Py_DECREF(level);
        logging_level = 50;
    }
    Py_DECREF(level);
}


void logger_debug(const char *fmt, ...)
{
    if (logging_level <= 10) {
        va_list ap;
        static PyObject *string = NULL;
        char message[150], msg[147];

        va_start(ap, fmt);
        vsprintf(msg, fmt, ap);
        va_end(ap);
        sprintf(message, "C! %s", msg);
        string = Py_BuildValue("s", message);

        PyObject_CallMethod(logger, "debug", "O", string);
        Py_DECREF(string);
    }
}


void logger_info(const char *fmt, ...)
{
    if (logging_level <= 20) {
        va_list ap;
        static PyObject *string = NULL;
        char message[150], msg[147];

        va_start(ap, fmt);
        vsprintf(msg, fmt, ap);
        va_end(ap);
        sprintf(message, "C! %s", msg);
        string = Py_BuildValue("s", message);

        PyObject_CallMethod(logger, "info", "O", string);
        Py_DECREF(string);
    }
}

void logger_warn(const char *fmt, ...)
{
    if (logging_level <= 30) {
        va_list ap;
        static PyObject *string = NULL;
        char message[150], msg[147];

        va_start(ap, fmt);
        vsprintf(msg, fmt, ap);
        va_end(ap);
        sprintf(message, "C! %s", msg);
        string = Py_BuildValue("s", message);

        PyObject_CallMethod(logger, "warning", "O", string);
        Py_DECREF(string);
    }
}


void logger_error(const char *fmt, ...)
{
    if (logging_level <= 40) {
        va_list ap;
        static PyObject *string = NULL;
        char message[150], msg[147];

        va_start(ap, fmt);
        vsprintf(msg, fmt, ap);
        va_end(ap);
        sprintf(message, "C! %s", msg);
        string = Py_BuildValue("s", message);

        PyObject_CallMethod(logger, "error", "O", string);
        Py_DECREF(string);
    }
}


void logger_critical(const char *fmt, ...)
{
    if (logging_level <= 50) {
        va_list ap;
        static PyObject *string = NULL;
        char message[150], msg[147];

        va_start(ap, fmt);
        vsprintf(msg, fmt, ap);
        va_end(ap);
        sprintf(message, "C! %s", msg);
        string = Py_BuildValue("s", message);

        PyObject_CallMethod(logger, "critical", "O", string);
        Py_DECREF(string);
    }
}
