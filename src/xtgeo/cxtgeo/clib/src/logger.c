#include "logger.h"

/* Logging levels: */

/* CRITICAL: 50 */
/* ERROR: 40 */
/* WARNING: 30 */
/* INFO: 20 */
/* DEBUG: 10 */
/* NOTSET: 0 */

static PyObject *XPYlogger;
static int XPYlogging_level = 0;

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

        XPYlogger = PyObject_CallMethod(logging, "getLogger", "s", "<cxtgeo>");


        /* Get the effective logging level */
        if (XPYlogging_level == 0) {
            PyObject *meth = PyObject_GetAttrString(XPYlogger, "getEffectiveLevel");
            PyObject *level = PyObject_CallFunctionObjArgs(meth, NULL);

            Py_DECREF(meth);

            if(level == NULL) {
                XPYlogging_level = 50;
            }
            else{
                XPYlogging_level = PyLong_AsLong(level);
            }
            if(PyErr_Occurred()){
                Py_DECREF(level);
                XPYlogging_level = 50;
            }
            Py_DECREF(level);
        }
        if (XPYlogging_level <= 20 && XPYlogging_level > 0) {
            logger_info("Logging in C using Python logger is activated ***");
        }
    }
}


void logger_debug(const char *fmt, ...)
{
    if (XPYlogging_level <= 10) {
        va_list ap;
        static PyObject *string = NULL;
        char message[150], msg[147];

        va_start(ap, fmt);
        vsprintf(msg, fmt, ap);
        va_end(ap);
        sprintf(message, "C! %s", msg);
        string = Py_BuildValue("s", message);

        PyObject_CallMethod(XPYlogger, "debug", "O", string);
        Py_DECREF(string);
    }
}


void logger_info(const char *fmt, ...)
{
    if (XPYlogging_level <= 20) {
        va_list ap;
        static PyObject *string = NULL;
        char message[150], msg[147];

        logger_init("");

        va_start(ap, fmt);
        vsprintf(msg, fmt, ap);
        va_end(ap);
        sprintf(message, "C! %s", msg);
        string = Py_BuildValue("s", message);

        PyObject_CallMethod(XPYlogger, "info", "O", string);
        Py_DECREF(string);
    }
}

void logger_warn(const char *fmt, ...)
{
    if (XPYlogging_level <= 30) {
        va_list ap;
        static PyObject *string = NULL;
        char message[150], msg[147];

        logger_init("");

        va_start(ap, fmt);
        vsprintf(msg, fmt, ap);
        va_end(ap);
        sprintf(message, "C! %s", msg);
        string = Py_BuildValue("s", message);

        PyObject_CallMethod(XPYlogger, "warning", "O", string);
        Py_DECREF(string);
    }
}


void logger_error(const char *fmt, ...)
{
    if (XPYlogging_level <= 40) {
        va_list ap;
        static PyObject *string = NULL;
        char message[150], msg[147];

        logger_init("");

        va_start(ap, fmt);
        vsprintf(msg, fmt, ap);
        va_end(ap);
        sprintf(message, "C! %s", msg);
        string = Py_BuildValue("s", message);

        PyObject_CallMethod(XPYlogger, "error", "O", string);
        Py_DECREF(string);
    }
}


void logger_critical(const char *fmt, ...)
{
    if (XPYlogging_level <= 50) {
        va_list ap;
        static PyObject *string = NULL;
        char message[150], msg[147];

        logger_init("");

        va_start(ap, fmt);
        vsprintf(msg, fmt, ap);
        va_end(ap);
        sprintf(message, "C! %s", msg);
        string = Py_BuildValue("s", message);

        PyObject_CallMethod(XPYlogger, "critical", "O", string);
        Py_DECREF(string);
    }
}
