#include <Python.h>

/* using pythons logging! */
void logger_init(const char *fname);
void logger_info(char *msg);
void logger_warn(char *msg);
