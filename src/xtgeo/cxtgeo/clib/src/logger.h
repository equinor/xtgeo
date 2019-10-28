#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

void logger_init(const char *funcname, const char *filename);

void logger_debug(int line, const char *fmt, ...);
void logger_info(int line, const char *fmt, ...);
void logger_warn(int line, const char *fmt, ...);
void logger_error(int line, const char *fmt, ...);
void logger_critical(int line, const char *fmt, ...);
