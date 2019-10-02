#include <Python.h>
#include <stdio.h>
#include <stdarg.h>

/* using pythons logging! */
void logger_init(const char *fname);

void logger_debug(const char *fmt, ...);
void logger_info(const char *fmt, ...);
void logger_warn(const char *fmt, ...);
void logger_error(const char *fmt, ...);
void logger_critical(const char *fmt, ...);
